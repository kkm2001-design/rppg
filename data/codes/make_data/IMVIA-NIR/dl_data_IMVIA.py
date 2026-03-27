import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np
from scipy import signal
import traceback

# ===== 경로 설정 =====
input_dir  = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/IMVIA-NIR_registration'
output_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/IMVIA-NIR_dl_data'
os.makedirs(output_dir, exist_ok=True)

window_size = 150
fs_video = 20.0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = fs / 2.0
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, data)

def z_norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

files = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5')])
print(f"총 {len(files)}개 파일 발견")

for fname in files:
    print('='*60)
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

            win_video = video_data[s:e].copy()  # (300, 64, 64)
            win_gt    = bvp_data[s:e].copy()    # (300,)

            try:
                # GT PPG 처리
                filter_gt  = butter_bandpass_filter(win_gt, 0.75, 2.5, fs_video, order=2)
                win_gt_norm = z_norm(filter_gt)

                valid_vid_list.append(win_video)
                valid_gt_list.append(win_gt_norm)

            except Exception as e:
                print(f'  Window {i} 에러: {e}')
                discarded.append(i)

        print(f"  유효 윈도우: {len(valid_vid_list)}개, 버린 윈도우: {len(discarded)}개")

        if len(valid_vid_list) > 0:
            video_array = np.stack(valid_vid_list)  # (W, 300, 64, 64)
            gt_array    = np.stack(valid_gt_list)   # (W, 300)

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