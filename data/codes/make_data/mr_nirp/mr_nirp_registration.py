import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from registration_algorithm import *


h5_dir = "/nasdata4/2yhy/rPPG_code/rPPG_datasets/h5_data/MR-NIRP_indoor"
output_dir ='/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/MR-nirp_registration_70'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thresh_angle = 70

files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])

for subject in files:
    print('='*80)
    print(f'처리중: {subject}')

    try:
        input_path = os.path.join(h5_dir, subject)
        output_path = os.path.join(output_dir, subject.replace('.h5', '_uv.h5'))

        # ===== [step1] 데이터 로드 =====
        print('-'*20, '[step1] 데이터 로드', '-'*20)

        with h5py.File(input_path, 'r') as f_r:
            raw_imgs = f_r['imgs'][:]      # (N, 128, 128, 1)
            bvp_data = f_r['bvp_data'][:]  # (N,)

        raw_imgs = raw_imgs.squeeze(-1)    # (N, 128, 128)
        num_frames_original = len(raw_imgs)
        print(f"  총 프레임: {num_frames_original}, shape: {raw_imgs[0].shape}")
        print(f"  bvp_data shape: {bvp_data.shape}")

        frames_xy = []
        for frame in tqdm(raw_imgs, desc='RGB 변환'):
            frames_xy.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
        frames_xy = np.array(frames_xy)

        print(f"  frames_xy shape: {frames_xy.shape}")
        print('-'*20, '[step1] 완료', '-'*20)

        # ===== [step2] 랜드마크 추출 =====
        print('-'*20, '[step2] 랜드마크 추출', '-'*20)

        detector = MediaPipeLandmarkDetector(
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )
        landmarks_xyz, valid_mask = detector(frames_xy)
        landmark_fail_rate = ((num_frames_original - valid_mask.sum()) / num_frames_original) * 100

        print(f"  landmarks shape: {landmarks_xyz.shape}")
        print(f"  랜드마크 성공률: {100 - landmark_fail_rate:.2f}%")
        print('-'*20, '[step2] 완료', '-'*20)

        # ===== [step3] UV 변환 =====
        print('-'*20, '[step3] UV domain transformation', '-'*20)

        valid_mask_np = valid_mask.detach().cpu()
        valid_indices = np.where(valid_mask_np.numpy())[0]
        num_valid_frames = len(valid_indices)

        landmarks_xy = landmarks_xyz[:, :, :2]
        landmarks_xy_valid = landmarks_xy[valid_mask]
        landmarks_xyz_valid = landmarks_xyz[valid_mask]

        warper = GPUPiecewiseAffineTransform(detector.landmarks_uv, device=device)
        batch_size = 64
        frames_uv_list = []

        with torch.no_grad():
            for j in tqdm(range(0, num_valid_frames, batch_size), desc="UV Warping"):
                idx = valid_indices[j : j + batch_size]
                b_frames = torch.from_numpy(frames_xy[idx]).to(device)
                b_lm = landmarks_xy_valid[j : j + batch_size]
                frames_uv_list.append(warper(b_frames, b_lm).cpu())
                if j % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()

        frames_uv = torch.cat(frames_uv_list, dim=0)
        print(f"  UV shape: {frames_uv.shape}")
        print('-'*20, '[step3] 완료', '-'*20)

        # ===== [step4] Angle Masking =====
        print('-'*20, '[step4] Angle Masking', '-'*20)

        masker = BatchUVAngleMasker(
            detector.mesh_faces,
            detector.landmarks_uv,
            device=device
        )
        frames_masked_list = []

        with torch.no_grad():
            for j in tqdm(range(0, frames_uv.shape[0], batch_size), desc="Masking"):
                b_uv = frames_uv[j : j + batch_size].to(device)
                b_lm = landmarks_xyz_valid[j : j + batch_size]
                if not b_lm.is_cuda:
                    b_lm = b_lm.to(device)
                frames_masked_list.append(
                    masker(b_uv, b_lm, degree_threshold=thresh_angle).cpu()
                )
                del b_uv, b_lm
                if j % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()

        frames_uv_masked = torch.cat(frames_masked_list, dim=0).numpy()
        print(f"  masked shape: {frames_uv_masked.shape}")
        print('-'*20, '[step4] 완료', '-'*20)

        # ===== [step5] 저장 =====
        print('-'*20, '[step5] 저장', '-'*20)

        frames_final = np.zeros((num_frames_original, 64, 64, 3), dtype=np.float32)
        frames_final[valid_mask_np] = frames_uv_masked
        result = frames_final[:, :, :, 0].astype(np.float32)

        with h5py.File(output_path, 'w') as f_w:
            dset = f_w.create_dataset(
                'nir_uv_data', data=result,
                compression='gzip', compression_opts=4
            )
            dset.attrs['fps'] = 30

            mask_dset = f_w.create_dataset(
                'valid_mask', data=valid_mask_np.numpy(),
                compression='gzip', compression_opts=4
            )
            mask_dset.attrs['fail_rate'] = landmark_fail_rate.item()

            f_w.create_dataset(
                'bvp_data', data=bvp_data,
                compression='gzip', compression_opts=4
            )

        print(f"  저장완료: {output_path}")
        print(f"  nir_uv_data : {result.shape}")
        print(f"  bvp_data    : {bvp_data.shape}")

        del frames_xy, raw_imgs, landmarks_xyz, frames_uv, frames_uv_masked, frames_final
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ {subject} 처리 실패")
        print(f"에러: {e}")
        traceback.print_exc()