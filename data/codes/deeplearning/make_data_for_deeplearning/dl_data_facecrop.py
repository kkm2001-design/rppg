import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import cv2
import numpy as np
import scipy.interpolate as interp
from tqdm import tqdm

sys.path.append('/home/son/orientation-uv-rppg/data/codes/deeplearning/utils')
from registration_algorithm import MediaPipeLandmarkDetector


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


def extract_crop_and_interp(
    video_windows,       # (N_windows, 900, H, W) 단채널
    output_size=(64, 64)
):
    clahe_obj = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    N_windows, window_size, H, W = video_windows.shape

    detector = MediaPipeLandmarkDetector(
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    )

    cropped_windows = []

    for win_i in tqdm(range(N_windows), desc='Windows'):
        win_frames = video_windows[win_i]  # (900, H, W)

        # ---- CLAHE + RGB 변환 (랜드마크 추출용) ----
        frames_clahe = []
        for f in win_frames:
            frame_clahe     = clahe_obj.apply(f)
            frame_clahe_rgb = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2RGB)
            frames_clahe.append(frame_clahe_rgb)
        frames_clahe = np.array(frames_clahe)  # (900, H, W, 3)

        # ---- 랜드마크 추출 ----
        landmarks_xyz, valid_mask = detector(frames_clahe)  # (900, 468, 3), (900,)

        landmarks_np = landmarks_xyz.cpu().numpy()  # (900, 468, 3)
        valid_np     = valid_mask.cpu().numpy()      # (900,) bool

        # ---- 크롭: 성공 → 크롭 + resize + 1채널, 실패 → 0 ----
        cropped_frames = np.zeros((window_size, output_size[0], output_size[1]), dtype=np.uint8)

        for f_i in range(window_size):
            if not valid_np[f_i]:
                continue  # 실패 프레임 → 0으로 유지

            lm_xy = landmarks_np[f_i, :, :2]  # (468, 2)
            x_min, x_max, y_min, y_max = get_face_bbox(lm_xy, H, W, margin=0.1)

            face = win_frames[f_i, y_min:y_max, x_min:x_max]  # (crop_h, crop_w) 단채널

            if face.size == 0:
                continue  # 크롭 실패 → 0 유지

            face_resized = cv2.resize(face, output_size)  # (64, 64)
            cropped_frames[f_i] = face_resized

        # ---- 크롭된 프레임 기준 PCHIP 보간 ----
        valid_idx = np.where(valid_np)[0]

        if len(valid_idx) > 1:
            cropped_frames = interp.pchip_interpolate(
                valid_idx,
                cropped_frames[valid_idx],      # (valid, 64, 64)
                np.arange(window_size),
                axis=0
            )  # (900, 64, 64)
            cropped_frames = np.clip(cropped_frames, 0, 255).astype(np.uint8)
        else:
            print(f'Window {win_i}: 유효 프레임 부족, 스킵')
            continue

        cropped_windows.append(cropped_frames)  # (900, 64, 64)

    return np.stack(cropped_windows)  # (N_windows, 900, 64, 64)


# ===== 설정 =====
people = ['A001_20250609', 'A002_20250609', 'A004_20250610', 'A005_20250611', 'A006_20250611',
          'A007_20250612', 'A008_20250612', 'B001_20250623', 'B002_20250623', 'B003_20250624',
          'B004_20250624', 'B005_20250625', 'B008_20250627', 'C001_20250701', 'C002_20250703', 'C003_20250702']

input_dir  = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration'
output_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped'
os.makedirs(output_dir, exist_ok=True)

# ===== 메인 루프 =====
for sub_idx in range(len(people)):
    print(f"\n--- Processing {people[sub_idx]} ---")

    # 이미 저장된 h5 로드
    input_path = os.path.join(input_dir, f'dataset_wo_registration_{people[sub_idx]}.h5')
    with h5py.File(input_path, 'r') as f:
        video_windows = f['video'][:]    # (N_windows, 900, H, W)
        final_gt_ppg  = f['gt_ppg'][:]  # (N_windows, 900)

    print(f'input shape : {video_windows.shape}')

    # 랜드마크 추출 + 크롭 + 보간
    final_raw_vid = extract_crop_and_interp(
        video_windows = video_windows,
        output_size   = (64, 64)
    )  # (N_windows, 900, 64, 64)

    print(f'output shape : {final_raw_vid.shape}')
    



    # 저장
    output_path = os.path.join(output_dir, f'dataset_wo_registration_cropped_{people[sub_idx]}.h5')
    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset(
            'video',
            data=final_raw_vid,
            chunks=(1, 900, 64, 64),
            compression='gzip',
            compression_opts=4
        )
        dset.attrs['subject_id'] = people[sub_idx]
        dset.attrs['fps'] = 30

        f.create_dataset(
            'gt_ppg',
            data=final_gt_ppg,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )

    print(f'{people[sub_idx]} 저장 완료 : {final_raw_vid.shape}')