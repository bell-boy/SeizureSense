import mne
import numpy as np
import torch
import os
import wfdb
from tqdm.notebook import tqdm
import einops
from multiprocessing import Pool

# TODO: Handle case where no seizure files are found
# TODO: Update Tests
class CHB_MIT_PAITENT(torch.utils.data.Dataset):
 
  def __init__(self, tok_len: int, ctx_len: int, path: str, sop: int, sph: int,  use_tok_dim: bool=False, regression: bool=False):
    '''
    Class for handling sequences of seizures. Accesses seizure data via chunking into small token sized units, and returning sequences of tokens. 
    Gaps are filled with zeros.

    Args:
      tok_len : int
        token duration in seconds
      ctx_len : int
        the length of sequence to be returned in tokens
      path : int 
        the directory in which seizure files are stored
      sop : int 
        Seizure Occurance Period in seconds, the maximum distance your context window can be from a seizure to be labeled preictal.
      sph : int
        Seizure Prediction Horizion in seconds, the minimum distnace your conext window can be from a seizure to be labeled preictal.

      use_tok_dim : bool
        (Default True) Most models at the time of writing don't want a token dimesion, they want the entire segment in one large block. This forces the 
        dataset to drop the token dimension
      regression : bool
        (Defualt False) set the dataset in regression mode
    '''
    self.tok_len = tok_len
    self.ctx_len = ctx_len
    self.path = path
    self.sop = sop
    self.eeg_raws = []
    self.seizure_periods = []
    self.use_tok_dim = use_tok_dim
    self.regression = regression

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

    Args:
      raw: mne.io.edf.edf.EDF_RAW

    Returns:
      (start_time: float, end_time: float)
    '''
    start = raw.info['meas_date'].timestamp() - self.baseline_time
    end = start + raw.times[-1]
    return start, end


  def __len__(self):
    total_time = int(self.final_time - self.baseline_time)
    if self.regression:
      return ((total_time - self.sop) // self.tok_len) - self.ctx_len + 1
    else:
      return (total_time // self.tok_len) - self.ctx_len + 1

  def get_label(self, int_start: int, int_end: int):
    '''
    returns 1 if a seizure occurs within the interval [start, end] 0 otherwise

    Args:
      int_start: int, interval start time in seconds
      int_end: int, interval end time in seconds
    '''
    for (start, end) in self.seizure_periods:
      if start >= int_start and start <= int_end:
        return 1
    return 0

  def get_next_seizure_time(self, time: int):
    '''
    find the time of the next seizure after the given time within the sop, else return inf

    Args:
      time: int
    ''' 
    for (start, end) in self.seizure_periods:
      if start >= time and start - time <= self.sop:
        return start
    return float('inf')

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
        uncat_segments.append(new_raw.copy().get_data())
      next_start, next_end = self.get_start_end(self.eeg_raws[idx+1])
      if end_time >= end and end_time <= next_start:
        break
    
    if not self.regression:
      label = self.get_label(end_time, end_time + self.sop)
    elif self.use_tok_dim:
      label = np.array([self.get_next_seizure_time(i + self.tok_len) - i for i in range(start_time, end_time, self.tok_len)])
    else:
      label = np.array(self.get_next_seizure_time(start_time + self.tok_len) - start_time)
    if len(uncat_segments) != 0:
      pre_pad = np.concatenate(uncat_segments, axis=-1)
      pad_len = int((end_time - start_time) * self.sample_rate - pre_pad.shape[-1])
    else:
      pre_pad = np.zeros((self.nchan, int((end_time - start_time) * self.sample_rate)))
      pad_len = 0
    if self.use_tok_dim:
      return (einops.rearrange(np.pad(pre_pad, ((0, 0), (0, pad_len))).astype(np.float32), "nchan (ctx_len tok_len) -> nchan ctx_len tok_len", ctx_len = self.ctx_len, nchan=self.nchan), label) 
    else:
      return (np.pad(pre_pad, ((0, 0), (0, pad_len))).astype(np.float32), label)

# TODO: Implement Undersampling
class FilteredCMP(CHB_MIT_PAITENT):
  def __init__(self, tok_len: int, ctx_len: int, path: str, sop: int, sph: int,  use_tok_dim: bool=False, regression: bool=False, balance_classes=False):
    '''
    A filtered version of CHB_MIT_PAITENT. Filters data out that is more than 2% sparse.

    Args:
      tok_len : int
        token duration in seconds
      ctx_len : int
        the length of sequence to be returned in tokens
      path : int 
        the directory in which seizure files are stored
      sop : int 
        Seizure Occurance Period in seconds, the maximum distance your context window can be from a seizure to be labeled preictal.
      sph : int
        Seizure Prediction Horizion in seconds, the minimum distnace your conext window can be from a seizure to be labeled preictal.
      use_tok_dim : bool
        (Default True) Most models at the time of writing don't want a token dimesion, they want the entire segment in one large block. This forces the 
        dataset to drop the token dimension
      regression : bool
        (Defualt False) set the dataset in regression mode
      balance_classes : bool
        (Default False) balance the number of positve and negative samples
    '''
    super().__init__(tok_len, ctx_len, path, sop, sph, use_tok_dim=use_tok_dim, regression=regression)
    self.filtered_idx_list = []
    self.temp = []
    self.neg_examples = []
    for i in tqdm(range(super().__len__())):
      data, label = super().__getitem__(i)
      if((np.abs(data) > 1e-8).astype(np.longlong).sum() / data.size > .98):
        self.filtered_idx_list.append(i)
    """
    n_total = 0
    self.mean_total = 0
    sum_squares_total = 0
    for idx in tqdm(self.filtered_idx_list):
      data, label = super().__getitem__(idx)
      n = data.size
      mean_subsample = np.mean(data)
      variance_subsample = np.var(data, ddof=0)  # Population variance

      # Update total mean and sum of squares
      n_total += n
      self.mean_total += n * mean_subsample
      sum_squares_total += (n - 1) * variance_subsample + n * mean_subsample**2

    # Final mean of the total sample
    self.mean_total /= n_total

    # Calculate total variance
    sum_squares_mean = n_total * self.mean_total**2
    self.variance_total = (sum_squares_total - sum_squares_mean) / n_total
    """

  def __len__(self):
    return len(self.filtered_idx_list)
  
  def __getitem__(self, idx):
    data, label = super().__getitem__(self.filtered_idx_list[idx])
    """
    data -= self.mean_total
    data /= np.sqrt(self.variance_total)
    """
    return data * 1e5, label
    