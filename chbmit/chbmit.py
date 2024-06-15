import mne
import numpy as np
import torch
import os
import wfdb
import einops

def stft(signal, window_len, hop_len, window_fn=np.hanning):
    '''
    Compute the STFT of a batch of signals with multiple batch dimensions

    Args:
      signal : ndarray
        A tensor of shape (..., n_samples)
      window_len : int
        window length in samples
      hop_len : int
        hop size in samples
      sample_rate : int
        sample rate in Hz, default 256
      window_fn : function
        windowing function, default np.hanning

    Returns:
      stft : ndarray
        the short time fourier transform of the given signal (Note that the negative frequencies are removed)
    '''
    if type(signal) is torch.Tensor:
      signal = signal.cpu()
    window = window_fn(window_len)
    
    n_windows = (signal.shape[-1] - window_len) // hop_len + 1
    
    stft_matrix = np.zeros(signal.shape[:-1] + (n_windows, window_len // 2 + 1), dtype=complex)
    
    for i in range(n_windows):
        start = i * hop_len
        end = start + window_len
        segment = signal[..., start:end] * window
        stft_matrix[..., i, :] = np.fft.rfft(segment, n=window_len)
    
    return stft_matrix

# TODO: Handle case where no seizure files are found
class CHB_MIT_PAITENT(torch.utils.data.Dataset):
 
  def __init__(self, tok_len: int, ctx_len: int, path: str, ppl: int=None, use_tok_dim: bool=False):
    '''
    Class for handling sequences of seizures. Accesses seizure data via chunking into small token sized units, and returning sequences of tokens. 
    Gaps are filled with zeros.

    Args:
      tok_len: token duration in seconds
      ctx_len: the length of sequence to be returned in tokens
      path: the directory in which seizure files are stored
      ppl: (defult None) preictal period length in seconds, the maximum distance your context window can be from a seizure to be labeled preictal. Set to None if you want the time
      of the next seizure instead
      use_tok_dim : boolean
        (Default True) Most models at the time of writing don't want a token dimesion, they want the entire segment in one large block. This forces the 
        dataset to drop the token dimension
    '''
    self.tok_len = tok_len
    self.ctx_len = ctx_len
    self.path = path
    self.ppl = ppl
    self.eeg_raws = []
    self.seizure_periods = []
    self.use_tok_dim = use_tok_dim

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
    find the next seizure after the given time

    Args:
      time: int
    ''' 
    for (start, end) in self.seizure_periods:
      if start >= time:
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
    
    label = self.get_label(end_time, end_time + self.ppl) if self.ppl is not None else self.get_next_seizure_time(end_time)
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

class STFT(torch.nn.Module):
  def __init__(self, window_len: int, hop_len: int, window_fn=np.hanning, log_scale=True, device=torch.device('cpu')):
    '''
    Module to compute the STFT of a batch of signals with multiple batch dimensions

    Init Args:
      window_len : int
        window length in samples
      hop_len : int
        hop size in samples
      sample_rate : int
        sample rate in Hz, default 256
      window_fn : function
        windowing function, default np.hanning
      device
        the device to which the STFT should return to 

    Forward Args:
      signal : ndarray
          A tensor of shape (..., n_samples)

    Returns:
      stft : ndarray
        the short time fourier transform of the given signals
    '''
    super().__init__()
    self.window_len = window_len
    self.hop_len = hop_len
    self.window_fn = window_fn
    self.log_scale = log_scale
    self.device = device
  
  def forward(self, signals):
    if not self.log_scale:
      return torch.Tensor(stft(signals, self.window_len, self.hop_len, window_fn=self.window_fn)).to(self.device)
    else:
      return torch.Tensor(np.log(np.abs(stft(signals, self.window_len, self.hop_len, window_fn=self.window_fn)) + 1e-8) * 10).to(self.device)