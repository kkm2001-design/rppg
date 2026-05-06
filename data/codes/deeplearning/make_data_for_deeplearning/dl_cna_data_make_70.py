import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks,welch
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

def bandpass_fft_soft_np(x, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    N = x.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_x = np.fft.rfft(x)
    mask = (1 / (1 + np.exp(-steepness * (freqs - low_hz)))) * \
           (1 / (1 + np.exp(-steepness * (high_hz - freqs))))
    return np.fft.irfft(fft_x * mask, n=N)
    
def z_norm(data):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

def process_video_and_gt_windows(video_data, gt_ppg, sub_valid, window_size=300, max_gap_threshold=30):
    n_windows = len(video_data) // window_size
    valid_vid_raw_list = []
    valid_gt_list = []
    discarded_indices = []

    for i in range(n_windows):
        s = i * window_size
        e = s + window_size

        win_video = video_data[s:e].copy()
        win_gt    = gt_ppg[s:e].copy()
        win_valid = sub_valid[s:e].copy()   

        # Gap 체크
        is_invalid = ~win_valid
        if np.any(is_invalid):
            condition  = np.concatenate(([0], is_invalid, [0]))
            idx        = np.flatnonzero(condition[1:] != condition[:-1])
            gap_lengths = idx[1::2] - idx[::2]
            max_gap    = gap_lengths.max() if len(gap_lengths) > 0 else 0
        else:
            max_gap = 0

        if max_gap >= max_gap_threshold:
            discarded_indices.append(i)
            continue

        try:
            # PCHIP 보간
            if max_gap > 0:
                valid_idx = np.where(win_valid)[0]
                if len(valid_idx) > 1:
                    win_video = interp.pchip_interpolate(valid_idx, win_video[valid_idx], np.arange(window_size), axis=0)
                else:
                    discarded_indices.append(i)
                    continue



            valid_vid_raw_list.append(win_video)
            valid_gt_list.append(win_gt)

        except Exception as e:
            print(f'  Window {i} Error: {e}')
            discarded_indices.append(i)

    return valid_vid_raw_list, valid_gt_list, discarded_indices

# ===== 설정 =====
fs_video = 30.0
fs_ppg   = 500.0

people = ['A001_20250609', 'A002_20250609', 'A004_20250610', 'A005_20250611', 'A006_20250611',
          'A007_20250612', 'A008_20250612', 'B001_20250623', 'B002_20250623', 'B003_20250624',
          'B004_20250624', 'B005_20250625', 'B008_20250627', 'C001_20250701', 'C002_20250703', 'C003_20250702']

people_ppg = sorted([k.replace('_20', '_') for k in people])

preprocess_result_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/uv_registration_CNA'
ppg_cut_dir           = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/ppg_cut_final'
final_save_dir        = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_len300'
os.makedirs(final_save_dir, exist_ok=True)

# ===== 메인 루프 =====
total_dict = {}

for sub_idx in range(len(people)):
    print(f"\n--- Processing {people[sub_idx]} ---")

    # 영상 로드
    h5_video_path = os.path.join(preprocess_result_dir, f'registration_video_[{people[sub_idx]}].h5')
    with h5py.File(h5_video_path, 'r') as f_v:
        video_data = f_v['nir_uv_data'][:]
        sub_valid  = f_v['valid_mask'][:]

    # GT PPG 로드 + 다운샘플링 (shift 없이 길이만 맞춤)
    gt_csv_path = os.path.join(ppg_cut_dir, f'{people_ppg[sub_idx]}.csv')
    ppg_cut = pd.read_csv(gt_csv_path).values.flatten()
    ppg_downsampled = signal.resample(ppg_cut, len(video_data))

    ppg_filt = bandpass_fft_soft_np(ppg_downsampled)
    ppg_norm = z_norm(ppg_filt)

    # 윈도우 처리
    valid_vid_raw_list, valid_gt_list, discarded = process_video_and_gt_windows(
        video_data=video_data,
        gt_ppg=ppg_norm,
        sub_valid=sub_valid,
        window_size=300,
        max_gap_threshold=30
    )
    nan_count = 0
    hr_list = []

    for i in range(len(valid_gt_list)):
        hr, _, _ = hr_fft(valid_gt_list[i], fs=30)
        if np.isnan(hr):
            nan_count += 1
            hr = 0.0  # nan 처리 → 0으로 채움
        hr_list.append(hr)

    hr_array = np.array(hr_list)  # (N,)



    # 이상치 제거 (0~1 범위 벗어나는 윈도우)
    indi_dict = {'num': 0, 'index': []}
    for w_idx in range(len(valid_vid_raw_list)):
        if np.max(valid_vid_raw_list[w_idx]) > 1.0 or np.min(valid_vid_raw_list[w_idx]) < 0.0:
            indi_dict['num'] += 1
            indi_dict['index'].append(w_idx)

    total_dict[people[sub_idx]] = indi_dict

    final_vid_list = [valid_vid_raw_list[w] for w in range(len(valid_vid_raw_list)) if w not in indi_dict['index']]
    final_gt_list  = [valid_gt_list[w]      for w in range(len(valid_gt_list))      if w not in indi_dict['index']]

    if len(final_vid_list) > 0:
        video_array = np.stack(final_vid_list)
        gt_array    = np.stack(final_gt_list)

        final_h5_path = os.path.join(final_save_dir, f'final_dataset_{people[sub_idx]}.h5')
        with h5py.File(final_h5_path, 'w') as f_out:
            f_out.create_dataset('video',  data=video_array, chunks=True, compression='gzip', compression_opts=4)
            f_out.create_dataset('gt_ppg', data=gt_array,    chunks=True, compression='gzip', compression_opts=4)
            f_out.create_dataset('hr', data=hr_array,    chunks=True, compression='gzip', compression_opts=4)

        print(f"  -> 이상치 {indi_dict['num']}개 제외")
        print(f"  -> 정상 윈도우 {len(final_vid_list)}개 저장: {final_h5_path}")
    else:
        print(f"  -> 경고: 저장할 윈도우 없음")