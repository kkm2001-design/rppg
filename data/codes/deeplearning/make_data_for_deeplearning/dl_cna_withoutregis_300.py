import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import cv2
import numpy as np
import pandas as pd
from scipy import signal
import scipy.interpolate as interp
from tqdm import tqdm

sys.path.append('/home/son/orientation-uv-rppg/data/codes/deeplearning')

# 그러면 아래처럼 utils.OOO 형태로 통일해서 가져올 수 있습니다.
from utils.registration_algorithm import MediaPipeLandmarkDetector
from utils.utils_sig import *
# (미리 정의해두신 get_face_bbox, MediaPipeLandmarkDetector 등은 그대로 사용)

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



def bandpass_fft_soft_np(x, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    N = x.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_x = np.fft.rfft(x)
    mask = (1 / (1 + np.exp(-steepness * (freqs - low_hz)))) * \
           (1 / (1 + np.exp(-steepness * (high_hz - freqs))))
    return np.fft.irfft(fft_x * mask, n=N)
    

def extract_and_filter_pipeline(
    video_data,        # (T, H, W) 전체 비디오
    gt_ppg,            # (T,) 전체 GT
    output_size=(64, 64),
    window_size=300,
    max_gap_threshold=30
):
    """
    1. 300프레임 단위로 자르기
    2. 랜드마크 추출 및 크롭 (실패 시 0으로 채움)
    3. 윈도우 내 연속 30프레임 이상 실패 시 해당 윈도우 폐기
    """
    T, H, W = video_data.shape
    n_windows = T // window_size
    
    detector = MediaPipeLandmarkDetector(
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    )
    clahe_obj = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))

    valid_vid_list = []
    valid_gt_list = []
    discarded_indices = []

    for i in tqdm(range(n_windows), desc='Processing Windows'):
        s = i * window_size
        e = s + window_size

        win_video = video_data[s:e]  # (300, H, W)
        win_gt    = gt_ppg[s:e]      # (300,)

        # ---- [Step 1] 랜드마크 추출 및 크롭 준비 ----
        frames_clahe = []
        for f in win_video:
            frame_clahe = clahe_obj.apply(f)
            frames_clahe.append(cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2RGB))
        frames_clahe = np.array(frames_clahe)

        landmarks_xyz, valid_mask = detector(frames_clahe)
        landmarks_np = landmarks_xyz.cpu().numpy()
        valid_np     = valid_mask.cpu().numpy()  # (300,) bool

        # ---- [Step 2] 연속 실패(Gap) 계산 및 윈도우 폐기 판별 ----
        is_invalid = ~valid_np
        if np.any(is_invalid):
            # 연속된 True(실패) 구간의 길이 계산
            condition   = np.concatenate(([0], is_invalid, [0]))
            idx         = np.flatnonzero(condition[1:] != condition[:-1])
            gap_lengths = idx[1::2] - idx[::2]
            max_gap     = gap_lengths.max() if len(gap_lengths) > 0 else 0
        else:
            max_gap = 0

        # 연속 30프레임 이상 랜드마크 추출 실패 시 윈도우 버림
        if max_gap >= max_gap_threshold:
            print(f'Window {i} 버림 (최대 연속 실패: {max_gap}프레임)')
            discarded_indices.append(i)
            continue

        # ---- [Step 1-2] 크롭 및 0으로 채우기 ----
        cropped_frames = np.zeros((window_size, output_size[0], output_size[1]), dtype=np.uint8)

        for f_i in range(window_size):
            if not valid_np[f_i]:
                continue  # 랜드마크 추출 안된 프레임은 0 배열로 유지

            lm_xy = landmarks_np[f_i, :, :2]
            x_min, x_max, y_min, y_max = get_face_bbox(lm_xy, H, W, margin=0.1)

            face = win_video[f_i, y_min:y_max, x_min:x_max]

            if face.size > 0:
                cropped_frames[f_i] = cv2.resize(face, output_size)

        cropped_frames = cropped_frames.astype(np.float32) / 255.0
        valid_idx = np.where(valid_np)[0]
        invalid_idx = np.where(~valid_np)[0]
     
        if len(invalid_idx) > 0 and len(valid_idx) > 1: # PCHIP은 최소 2개의 정상 프레임 필요
        # 1. 300장 전체가 아닌, 빵꾸난 곳(invalid_idx)만 PCHIP으로 계산!
            interpolated_images = interp.pchip_interpolate(valid_idx, cropped_frames[valid_idx], invalid_idx, axis=0)
            cropped_frames[invalid_idx] = interpolated_images
        valid_vid_list.append(cropped_frames)
        valid_gt_list.append(win_gt)

    final_vid = np.stack(valid_vid_list)
    final_gt = np.stack(valid_gt_list)

    return final_vid, final_gt, discarded_indices


fs_video = 30.0
fs_ppg   = 500.0

people = ['A001_20250609', 'A002_20250609', 'A004_20250610', 'A005_20250611', 'A006_20250611',
          'A007_20250612', 'A008_20250612', 'B001_20250623', 'B002_20250623', 'B003_20250624',
          'B004_20250624', 'B005_20250625', 'B008_20250627', 'C001_20250701', 'C002_20250703', 'C003_20250702']

people_ppg = sorted([k.replace('_20', '_') for k in people])

preprocess_result_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/raw_video_CNA'
ppg_cut_dir           = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/ppg_cut_final'
final_save_dir        = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped_len300'
os.makedirs(final_save_dir, exist_ok=True)


total_dict = {}

for sub_idx in range(len(people)):
    print(f"\n--- Processing {people[sub_idx]} ---")

    # 영상 로드
    h5_video_path = os.path.join(preprocess_result_dir, f'raw_video_[{people[sub_idx]}].h5')
    with h5py.File(h5_video_path, 'r') as f_v:
        video_data = f_v['video'][:] 


    # GT PPG 로드 + 다운샘플링 (shift 없이 길이만 맞춤)
    gt_csv_path = os.path.join(ppg_cut_dir, f'{people_ppg[sub_idx]}.csv')
    ppg_cut = pd.read_csv(gt_csv_path).values.flatten()
    ppg_downsampled = signal.resample(ppg_cut, len(video_data))

    ppg_filt = bandpass_fft_soft_np(ppg_downsampled)
    ppg_norm = z_norm(ppg_filt)

    # 윈도우 처리
    valid_vid_raw_list, valid_gt_list, discarded = extract_and_filter_pipeline(
        video_data=video_data,
        gt_ppg=ppg_norm,
        output_size=(64, 64),
        window_size=300,
        max_gap_threshold=30
    )
    nan_count = 0
    hr_list = []

    # 이상치 제거 (0~1 범위 벗어나는 윈도우)
    indi_dict = {'num': 0, 'index': []}
    for w_idx in range(len(valid_vid_raw_list)):
        if np.max(valid_vid_raw_list[w_idx]) > 1.0 or np.min(valid_vid_raw_list[w_idx]) < 0.0:
            indi_dict['num'] += 1
            indi_dict['index'].append(w_idx)

    total_dict[people[sub_idx]] = indi_dict

    final_vid_list = [valid_vid_raw_list[w] for w in range(len(valid_vid_raw_list)) if w not in indi_dict['index']]
    final_gt_list  = [valid_gt_list[w]      for w in range(len(valid_gt_list))      if w not in indi_dict['index']]

    for i in range(len(final_gt_list)):
        hr, _, _ = hr_fft(final_gt_list[i], fs=30)
        if np.isnan(hr):
            nan_count += 1
            hr = 0.0  # nan 처리 → 0으로 채움
        hr_list.append(hr)

    hr_array = np.array(hr_list)  # (N,)    

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