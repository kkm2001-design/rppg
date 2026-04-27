import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np
def bandpass_fft_soft_np(x, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    N = x.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_x = np.fft.rfft(x)
    mask = (1 / (1 + np.exp(-steepness * (freqs - low_hz)))) * \
           (1 / (1 + np.exp(-steepness * (high_hz - freqs))))
    return np.fft.irfft(fft_x * mask, n=N)

DATA_DIR = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_driving_withoutoutlier_len300_gtsubjectznorm'

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])

for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    with h5py.File(fpath, 'r+') as f:
        gt = f['gt_ppg'][:]            # (N_windows, 300)
        gt_filtered = bandpass_fft_soft_np(gt)  # axis=-1 기본 적용
        f['gt_ppg'][...] = gt_filtered
    print(f'{fname}: {gt.shape} done')