from chbmit import chbmit
import os
import numpy as np
import mne
import einops

raw_1 = mne.io.read_raw_edf(os.path.dirname(os.path.abspath(__file__)) + "/test_data/chb01_01.edf")
raw_2 = mne.io.read_raw_edf(os.path.dirname(os.path.abspath(__file__)) + "/test_data/chb01_02.edf")
raw_3 = mne.io.read_raw_edf(os.path.dirname(os.path.abspath(__file__)) + "/test_data/chb01_03.edf")

data = chbmit.CHB_MIT_PAITENT(10, 10, os.path.dirname(os.path.abspath(__file__)) + "/test_data")

def test_seizures():
  assert data.seizure_periods == [(7_210.0 + 2996, 7_210.0 + 3036)]

def test_correct_shape():
  for i in range(len(data)):
    assert data[i][0].shape == (23, 10, 2560)

def test_start_in_end_in():
  assert data[1][0].shape == (23, 10, 2560)
  mock = einops.rearrange(raw_1.copy().crop(tmin=10, tmax=110, include_tmax=False).get_data(), "nchan (ctx_len tok_len) -> nchan ctx_len tok_len", ctx_len = data.ctx_len, nchan=data.nchan)
  assert np.allclose(data[1][0], mock)

def test_start_out_end_in():
  assert data[360][0].shape == (23, 10, 2560)
  mock = einops.rearrange(np.pad(raw_2.copy().crop(tmax=3700 - 3603.0, include_tmax=False).get_data(), ((0, 0), (768, 0))), "nchan (ctx_len tok_len) -> nchan ctx_len tok_len", ctx_len = data.ctx_len, nchan=data.nchan)
  assert np.allclose(data[360][0], mock)