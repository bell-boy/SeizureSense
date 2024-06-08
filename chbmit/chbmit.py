import mne
import numpy as np
import torch
import os
import wfdb
import einops

class CHB_MIT_PAITENT(torch.utils.data.Dataset):
 
  def __init__(self, tok_len: int, ctx_len: int, path: str):
    '''
    Class for handling sequences of seizures. Accesses seizure data via chunking into small token sized units, and returning sequences of tokens. 
    Gaps are filled with zeros.

    Args:
      tok_len: token duration in seconds
      ctx_len: the length of sequence to be returned in tokens
      path: the directory in which seizure files are stored
    '''
    self.tok_len = tok_len
    self.ctx_len = ctx_len
    self.path = path
    self.eeg_raws = []
    self.seizure_periods = []

    # Get all the EEG raws in the path directory and sort them by time stamp
    # Load raws
    for file in os.listdir(self.path):
      if file.endswith(".edf"):
        full_path = os.path.join(self.path, file)
        raw = mne.io.read_raw_edf(full_path, verbose='ERROR')
        self.sample_rate = raw.info['sfreq']
        self.nchan = raw.info['nchan']
        timestamp = raw.info['meas_date'].timestamp()
        self.eeg_raws.append((raw, timestamp))

        # check to see if this file contains a seizure
        if os.path.exists(full_path + ".seizures"):
          ann = wfdb.io.rdann(full_path, "seizures")
          self.seizure_periods.append(tuple(ann.sample / self.sample_rate + timestamp))

    # Sort by Time Start
    self.eeg_raws.sort(key=lambda x: x[1])
    self.eeg_raws = [raw[0] for raw in self.eeg_raws]

    # Gather seizure period times & bookend times
    self.baseline_time = self.eeg_raws[0].info['meas_date'].timestamp()
    self.final_time = self.eeg_raws[-1].info['meas_date'].timestamp() + self.eeg_raws[-1].times[-1]

    self.seizure_periods = [(start - self.baseline_time, end - self.baseline_time) for start, end in self.seizure_periods]
    self.seizure_periods.sort(key=lambda x: x[0])


  def get_start_end(self, raw):
    '''
    Helper function to get the start and end time of the raw relative to the paitnet baseline
    args:
      raw: mne.io.edf.edf.EDF_RAW
    returns:
      (start_time: float, end_time: float)
    '''
    start = raw.info['meas_date'].timestamp() - self.baseline_time
    end = start + raw.times[-1]
    return start, end


  def __len__(self):
    total_time = int(self.final_time - self.baseline_time)
    return (total_time // self.tok_len) - self.ctx_len + 1

  def __getitem__(self, idx):
    start_time = idx * self.tok_len
    end_time = start_time + self.tok_len * self.ctx_len 

    # This mess of code is how we pad gaps with zeros
    # It works by concatiating from the start to the end
    # I do it this way to prevent having to load all the files into memory at once
    uncat_segments = []
    started = False
    for idx, raw in enumerate(self.eeg_raws):
      new_raw = None
      start, end = self.get_start_end(raw)

      # This code is to test if the current start point is in a gap OR we're picking up a segment that comes after a gap.
      if idx != 0:
          prev_start, prev_end = self.get_start_end(self.eeg_raws[idx-1])
          if not started and start_time >= prev_end and start_time <= start:
            uncat_segments.append(np.zeros((int(raw.info['nchan']), int(int(start - start_time) * self.sample_rate))))
            started = True
          elif started:
            uncat_segments.append(np.zeros((int(raw.info['nchan']), int(int(start - prev_end) * self.sample_rate))))

      if not started:
        if start_time >= start and start_time <= end:
          new_raw = raw.copy().crop(tmin=start_time - start,)
          started = True
          # Dealing with the edge case that the end point is in the same file, or just right after it
          if end_time >= start and end_time <= end:
            new_raw = raw.copy().crop(tmin=start_time - start, tmax=end_time - start, include_tmax=False)
            uncat_segments.append(new_raw.get_data())
            break
        

      if started:
        if end_time >= start and end_time <= end:
          new_raw = raw.copy().crop(tmax=end_time-start, include_tmax=False)
          uncat_segments.append(new_raw.get_data())
          break

      
      if started:
        uncat_segments.append(raw.copy().get_data())
      next_start, next_end = self.get_start_end(self.eeg_raws[idx+1])
      if end_time >= end and end_time <= next_start:
        break
   
    pre_pad = np.concatenate(uncat_segments, axis=-1)
    pad_len = int((end_time - start_time) * self.sample_rate - pre_pad.shape[-1])
    return einops.rearrange(np.pad(pre_pad, ((0, 0), (0, pad_len))), "nchan (ctx_len tok_len) -> nchan ctx_len tok_len", ctx_len = self.ctx_len, nchan=self.nchan)

