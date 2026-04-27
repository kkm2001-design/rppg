import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks,welch
from scipy.fft import fft
sys.path.append('/home/son/orientation-uv-rppg/data/codes/deeplearning/utils')
from registration_algorithm import MediaPipeLandmarkDetector


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




def get_hr(ppg, rppg, window_size=900, lowcut=0.75, highcut=2.5) : 
    
    ae_list = []
    rmse_list = []
    nan_list = []
    ppg_list = []
    rppg_list = []

    for i in list(ppg.keys()) :

        # hr_fft에 lowcut, highcut 인자 전달
        hr_ppg, psd_ppg, axis_ppg = hr_fft(ppg[i], fs=30, lowcut=lowcut, highcut=highcut)
        hr_rppg, psd_rppg, axis_rppg = hr_fft(rppg[i], fs=30, lowcut=lowcut, highcut=highcut)

        if np.isnan(hr_ppg) or np.isnan(hr_rppg) : 
            nan_list.append(i)
            continue 
        
        ae = abs(hr_ppg - hr_rppg)
        ae_list.append(ae)

        rmse = (hr_ppg - hr_rppg)**2
        rmse_list.append(rmse)

        ppg_list.append(hr_ppg)
        rppg_list.append(hr_rppg)

    mae = np.mean(ae_list)
    rmse = np.sqrt(np.mean(rmse_list))
    r = np.corrcoef(ppg_list, rppg_list)[0,1]

    return mae, nan_list , rmse , r

def z_norm(data):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

# ===== bandpass filter (기존 registration 데이터와 동일) =====
def bandpass_fft_soft_np(x, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    N = x.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_x = np.fft.rfft(x)
    mask = (1 / (1 + np.exp(-steepness * (freqs - low_hz)))) * \
           (1 / (1 + np.exp(-steepness * (high_hz - freqs))))
    return np.fft.irfft(fft_x * mask, n=N)


# ===== landmark 기반 bbox 추출 =====
def get_face_bbox(landmark_xy, h, w, margin=0.1):
    x_min = int(np.clip(landmark_xy[:, 0].min(), 0, w))
    x_max = int(np.clip(landmark_xy[:, 0].max(), 0, w))
    y_min = int(np.clip(landmark_xy[:, 1].min(), 0, h))
    y_max = int(np.clip(landmark_xy[:, 1].max(), 0, h))

    x_margin = int((x_max - x_min) * margin)
    y_margin = int((y_max - y_min) * margin)

    x_min = max(0, x_min - x_margin)
    x_max = min(w, x_max + x_margin)
    y_min = max(0, y_min - y_margin)
    y_max = min(h, y_max + y_margin)

    return x_min, x_max, y_min, y_max


# ===== crop + resize (CLAHE 없음, 보간 없음) =====
def crop_windows(video_windows, output_size=(128, 128)):
    """
    video_windows: (N_windows, 300, H, W) uint8 단채널
    return: (N_windows, 300, 64, 64) uint8
    """
    N_windows, window_size, H, W = video_windows.shape

    detector = MediaPipeLandmarkDetector(
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    )

    cropped_windows = []

    for win_i in tqdm(range(N_windows), desc='Cropping windows'):
        win_frames = video_windows[win_i]  # (300, H, W)

        # MediaPipe는 RGB 입력 필요 → grayscale을 3ch로 변환
        frames_rgb = np.stack([
            cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in win_frames
        ])  # (300, H, W, 3)

        # 랜드마크 추출
        landmarks_xyz, valid_mask = detector(frames_rgb)  # (300, 468, 3), (300,)
        landmarks_np = landmarks_xyz.cpu().numpy()
        valid_np = valid_mask.cpu().numpy()

        # 크롭: 실패 프레임 → 0 유지
        cropped_frames = np.zeros(
            (window_size, output_size[0], output_size[1]), dtype=np.uint8
        )

        for f_i in range(window_size):
            if not valid_np[f_i]:
                continue

            lm_xy = landmarks_np[f_i, :, :2]
            x_min, x_max, y_min, y_max = get_face_bbox(lm_xy, H, W, margin=0.1)
            face = win_frames[f_i, y_min:y_max, x_min:x_max]

            if face.size == 0:
                continue

            cropped_frames[f_i] = cv2.resize(face, output_size)

        cropped_windows.append(cropped_frames)

    return np.stack(cropped_windows)  # (N_windows, 300, 64, 64)


# ===== 설정 =====
input_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/h5_data/MR-NIRP_driving_motion'
output_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/driving_cropped_dl_data'
os.makedirs(output_dir, exist_ok=True)

fs = 30
window_size = 300

files = sorted([os.path.join(input_dir,s) for s in os.listdir(input_dir) if s.endswith('garage_large_motion_975.h5') ])

#h5_files = sorted(glob.glob(os.path.join(input_dir, '*.h5')))
print(f'Found {len(files)} h5 files')

# ===== 메인 루프 =====
for h5_path in files:
    fname = os.path.basename(h5_path)
    print(f'\n--- Processing {fname} ---')

    # 1. 로드
    with h5py.File(h5_path, 'r') as f:
        imgs = f['imgs'][:]        # (T, 128, 128, 1)
        bvp = f['bvp_data'][:]     # (T,)

    bvp = bandpass_fft_soft_np(bvp)
    bvp = z_norm(bvp)

    imgs = imgs.squeeze(-1)         # (T, 128, 128)
    T = imgs.shape[0]
    print(f'Total frames: {T}')

    # 2. 300프레임 windowing (non-overlapping)
    n_windows = T // window_size
    imgs_windowed = imgs[:n_windows * window_size].reshape(n_windows, window_size, 128, 128)
    bvp_windowed = bvp[:n_windows * window_size].reshape(n_windows, window_size)

    print(f'Windows: {n_windows} (remainder {T - n_windows * window_size} frames dropped)')

    nan_count = 0
    hr_list = []
    
    for i in range(bvp_windowed.shape[0]):
        hr, _, _ = hr_fft(bvp_windowed[i], fs=30)
        if np.isnan(hr):
            nan_count += 1
            hr = 0.0  # nan 처리 → 0으로 채움
        hr_list.append(hr)

    hr_array = np.array(hr_list)  # (N,)
    

    # 4. MediaPipe crop → 64×64
    cropped = crop_windows(imgs_windowed, output_size=(128, 128))
    # (n_windows, 300, 64, 64)

    print(f'Cropped shape: {cropped.shape}')

    # 5. 저장
    out_name = fname.replace('.h5', '_dldata.h5')
    output_path = os.path.join(output_dir, out_name)

    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset(
            'video',
            data=cropped,
            chunks=(1, window_size, 128, 128),
            compression='gzip',
            compression_opts=4
        )
        dset.attrs['fps'] = fs
        dset.attrs['source'] = fname

        f.create_dataset(
            'gt_ppg',
            data=bvp_windowed,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )


        f.create_dataset('hr', data=hr_array,    chunks=True, compression='gzip', compression_opts=4)

    print(f'Saved: {output_path} | video {cropped.shape}, ppg {bvp_windowed.shape}')

print('\nDone.')