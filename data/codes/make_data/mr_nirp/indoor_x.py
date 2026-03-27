import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np
from scipy import signal
import traceback

input_dir  = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/MR-nirp_indoor_registration_70'
output_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/mrNIRP_dl_datax'
os.makedirs(output_dir, exist_ok=True)

window_size = 300
fs_video    = 30

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = fs / 2.0
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, data)

def z_norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# ── 추가된 함수들 ──────────────────────────────────────────

def diff_normalize_video(data):
    """
    (N, H, W) → (N, H, W)
    1) (I[t+1]-I[t]) / (I[t+1]+I[t]+ε)
    2) 전체 std로 normalize
    3) ±3 sigma outlier clipping  (rPPG-Toolbox 방식)
    4) 마지막 프레임 zero-padding (길이 유지)
    """
    n, h, w = data.shape
    diff = np.zeros((n - 1, h, w), dtype=np.float32)

    for t in range(n - 1):
        num = data[t+1].astype(np.float32) - data[t].astype(np.float32)
        den = data[t+1].astype(np.float32) + data[t].astype(np.float32) + 1e-7
        diff[t] = num / den

    diff = diff / (np.std(diff) + 1e-8)         # std normalize
    diff = np.clip(diff, -3.0, 3.0)             # ±3σ outlier clipping
    diff[np.isnan(diff)] = 0

    # 길이를 원본 N으로 맞추기 위해 zero-padding
    padding = np.zeros((1, h, w), dtype=np.float32)
    diff = np.concatenate([diff, padding], axis=0)  # (N, H, W)
    return diff

def diff_normalize_label(label):
    """
    label: (N,)
    np.diff → std normalize → zero-padding
    """
    d = np.diff(label.astype(np.float32))
    d = d / (np.std(d) + 1e-8)
    d[np.isnan(d)] = 0
    d = np.append(d, 0.0)  # (N,) 길이 유지
    return d

# ─────────────────────────────────────────────────────────

files = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5')])
print(f"총 {len(files)}개 파일 발견")

for fname in files:
    print('=' * 60)
    print(f'처리중: {fname}')
    try:
        with h5py.File(os.path.join(input_dir, fname), 'r') as f:
            video_data = f['nir_uv_data'][:]  # (N, 64, 64)
            bvp_data   = f['bvp_data'][:]     # (N,)

        n_samples = len(video_data)
        n_windows = n_samples // window_size
        print(f"  총 프레임: {n_samples}, 윈도우 수: {n_windows}")

        valid_vid_list = []
        valid_gt_list  = []
        discarded      = []

        for i in range(n_windows):
            s = i * window_size
            e = s + window_size
            win_video = video_data[s:e].copy()  # (150, 64, 64)
            win_gt    = bvp_data[s:e].copy()    # (150,)

            try:
                # ── Video: diff_normalize (outlier clipping 포함) ──
                win_vid_proc = diff_normalize_video(win_video)  # (150, 64, 64)

                # ── GT PPG: bandpass → diff_normalize ──

                win_gt_proc  = diff_normalize_label(win_gt)  # (150,)
                filter_gt    = butter_bandpass_filter(win_gt_proc, 0.75, 2.5, fs_video, order=2)

                valid_vid_list.append(win_vid_proc)
                valid_gt_list.append(win_gt_proc)

            except Exception as e:
                print(f'  Window {i} 에러: {e}')
                discarded.append(i)

        print(f"  유효 윈도우: {len(valid_vid_list)}개, 버린 윈도우: {len(discarded)}개")

        if len(valid_vid_list) > 0:
            video_array = np.stack(valid_vid_list)  # (W, 150, 64, 64)
            gt_array    = np.stack(valid_gt_list)   # (W, 150)

            out_path = os.path.join(output_dir, fname.replace('.h5', '_windowed.h5'))
            with h5py.File(out_path, 'w') as f_out:
                f_out.create_dataset('video',  data=video_array, compression='gzip', compression_opts=4)
                f_out.create_dataset('gt_ppg', data=gt_array,    compression='gzip', compression_opts=4)

            print(f"  video shape : {video_array.shape}")
            print(f"  gt_ppg shape: {gt_array.shape}")
            print(f"  ✅ 저장완료: {out_path}")
        else:
            print(f"  ⚠️ 저장할 윈도우 없음")

    except Exception as e:
        print(f"❌ {fname} 실패: {e}")
        traceback.print_exc()