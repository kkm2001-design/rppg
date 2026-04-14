import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np
import glob
from scipy import signal
from scipy.fft import fft

def hr_fft(sig, fs, harmonics_removal=True, lowcut=0.66, highcut=2.5):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0]) # 양사이드 0처리
    sig_f = np.abs(fft(sig))
    
    low_idx = np.round(lowcut / fs * sig.shape[0]).astype('int') 
    high_idx = np.round(highcut / fs * sig.shape[0]).astype('int') 
    
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    
    if len(peak_idx) == 0 : 
        x_hr = np.arange(len(sig))/len(sig)*fs*60
        return  np.nan , sig_f_original, x_hr

    if len(peak_idx) == 1 :
        hr = (peak_idx[0] * fs) * 60 / sig.shape[0]
        x_hr = np.arange(len(sig))/len(sig)*fs*60
        return hr, sig_f_original, x_hr
        
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60

    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr


DATA_DIR = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_len300'

h5_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))

for fpath in h5_files:
    with h5py.File(fpath, 'a') as f:
        if 'hr' in f:
            print(f'[SKIP] {os.path.basename(fpath)}')
            continue

        ppg_all = f['gt_ppg'][:]  # (N, 900)
        hr_list = []
        nan_count = 0

        for i in range(ppg_all.shape[0]):
            hr, _, _ = hr_fft(ppg_all[i], fs=30)
            if np.isnan(hr):
                nan_count += 1
                hr = 0.0  # nan 처리 → 0으로 채움
            hr_list.append(hr)

        hr_array = np.array(hr_list)  # (N,)
        f.create_dataset('hr', data=hr_array)
        print(f'[DONE] {os.path.basename(fpath)} | shape: {hr_array.shape} | nan→0: {nan_count}개')