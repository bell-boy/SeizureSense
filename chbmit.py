import mne
import datetime
import numpy as np
import torch

class CHB_MIT_PAITENT(torch.utils.data.Dataset):
  def __init__(self, path: str, num_files: int, sample_rate: int, seizure_intervals: list[tuple[int, int]]):
    super().__init__()
    self.path = path
    self.num_files = num_files
    self.sample_rate = sample_rate
    self.eeg_raws = []
    self.seizure_intervals = seizure_intervals

    for i in range(1, self.num_files + 1):
      try:
        self.eeg_raws.append(mne.io.read_raw_edf(f"{self.path}{i:02}.edf"))
      except:
        continue

    self.baseline_time = self.eeg_raws[0].info['meas_date'].timestamp()
    self.duration = self.eeg_raws[-1].info['meas_date'].timestamp() - self.eeg_raws[0].info['meas_date'].timestamp()
    self.duration += self.eeg_raws[-1].times[-1] - self.eeg_raws[-1].times[0]

  def __len__(self):
    return int(self.duration // (10 * 60) - 4)

  def get_interval(self, req_start_time: int, req_end_time: int):
    raw_1: mne.io.edf.edf.RawEDF = None
    raw_2: mne.io.edf.edf.RawEDF = None
    idx_1: int = None
    idx_2: int = None

    normal_length = (req_end_time - req_start_time) * self.sample_rate

    for raw_idx, raw in enumerate(self.eeg_raws):
      start = raw.info['meas_date'].timestamp() - self.baseline_time
      end = start + raw.times[-1] - raw.times[0]
      if req_start_time >= start and req_start_time <= end:
        raw_1 = raw.copy()
        idx_1 = raw_idx
        req_start_time -= start

      if req_end_time >= start and req_end_time <= end:
        raw_2 = raw.copy()
        idx_2 = raw_idx
        req_end_time -= start

    if idx_1 == idx_2:
      return raw_1.crop(tmin=req_start_time, tmax=req_end_time, include_tmax=False).get_data()

    elif raw_1 == None:
      given_data = raw_2.crop(tmax=req_end_time, include_tmax=False).get_data()
      zero_pad = np.zeros((23, int(normal_length - given_data.shape[-1])))
      return np.concatenate( (zero_pad, given_data), axis=-1)

    elif raw_2 == None:
      given_data = raw_1.crop(tmin=req_start_time, include_tmax=False).get_data()
      zero_pad = np.zeros((23, int(normal_length - given_data.shape[-1])))
      return np.concatenate( (given_data, zero_pad), axis=-1)

    else:
      part_1 = raw_1.crop(tmin=req_start_time, include_tmax=False).get_data()
      part_2 = raw_2.crop(tmax=req_end_time, include_tmax=False).get_data()
      zero_pad = np.zeros((23, int(normal_length - part_1.shape[-1] - part_2.shape[-1])))
      return np.concatenate( (part_1, zero_pad, part_2), axis=-1)

  def has_seizure(self, req_start_time: int, req_end_time: int):
    for (start, end) in self.seizure_intervals:
      if start >= req_start_time and start <= req_end_time:
        return np.array([1])
      else:
        return np.array([0])

  def __getitem__(self, idx: int):

    start_time = idx * (10 * 60)
    end_time = start_time + (10 * 60)

    return self.get_interval(start_time, end_time), self.has_seizure(end_time + (10 * 60), end_time + (30 * 60))
