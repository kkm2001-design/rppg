import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import cv2
import torch
import numpy as np
from tqdm import tqdm
import traceback
from registration_algorithm import MediaPipeLandmarkDetector, GPUPiecewiseAffineTransform, BatchUVAngleMasker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thresh_angle = 70.0
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))

save_folder_raw      = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/raw_video'
preprocess_result_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/uv_registration_CNA'


people = ['A001_20250609','A002_20250609','A004_20250610','A005_20250611','A006_20250611','A007_20250612','A008_20250612','B001_20250623',
          'B002_20250623','B003_20250624','B004_20250624','B005_20250625','B008_20250627','C001_20250701','C002_20250703','C003_20250702']

raw_files = sorted([f for f in os.listdir(save_folder_raw) if f.endswith('.h5')])
print(f"총 {len(raw_files)}개 파일 발견")
print(raw_files)

for raw_fname in raw_files:
    print('='*80)
    print(f'처리중: {raw_fname}')

    try:
        h_dir = os.path.join(save_folder_raw, raw_fname)

        # ===== [step1] raw_video h5 로드 =====
        print('-'*20, '[step1] 데이터 로드', '-'*20)

        frames_xy = []
        frames_clahe_list = []

        with h5py.File(h_dir, 'r') as f_r:
            print(f"  Video shape: {f_r['video'].shape}")
            for frame in tqdm(f_r['video'], desc='RGB 변환'):
                frames_xy.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
            for frame in tqdm(f_r['video'], desc='CLAHE 변환'):
                frame_clahe = clahe.apply(frame)
                frames_clahe_list.append(cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2RGB))

        frames_xy = np.array(frames_xy)
        frames_clahe = np.array(frames_clahe_list)
        del frames_clahe_list

        num_frames_original = len(frames_xy)
        print(f"  총 프레임: {num_frames_original}, shape: {frames_xy[0].shape}")
        print('-'*20, '[step1] 완료', '-'*20)

        # ===== [step2] 랜드마크 추출 =====
        print('-'*20, '[step2] 랜드마크 추출', '-'*20)

        detector = MediaPipeLandmarkDetector(
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )
        landmarks_xyz, valid_mask = detector(frames_clahe)
        landmark_fail_rate = ((num_frames_original - valid_mask.sum()) / num_frames_original) * 100

        print(f"  landmarks shape: {landmarks_xyz.shape}")
        print(f"  랜드마크 성공률: {100 - landmark_fail_rate:.2f}%")
        print('-'*20, '[step2] 완료', '-'*20)

        # ===== [step3] UV 변환 =====
        print('-'*20, '[step3] UV domain transformation', '-'*20)

        valid_mask = valid_mask.detach().cpu()
        valid_indices = np.where(valid_mask.numpy())[0]
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

        frames_uv_masked = torch.cat(frames_masked_list, dim=0).detach().cpu().numpy()
        print(f"  masked shape: {frames_uv_masked.shape}")
        print('-'*20, '[step4] 완료', '-'*20)

        # ===== [step5] 저장 =====
        print('-'*20, '[step5] 저장', '-'*20)

        frames_uv_masked_final = np.zeros((num_frames_original, 64, 64, 3), dtype=np.float32)
        frames_uv_masked_final[valid_mask] = frames_uv_masked
        result_for_saving = frames_uv_masked_final[:, :, :, 0].astype(np.float32)

        # raw_video → registration_video 로 파일명 변환
        out_fname = raw_fname.replace('raw_video', 'registration_video')
        final_dir = os.path.join(preprocess_result_dir, out_fname)

        mb_size = result_for_saving.nbytes / (1024 ** 2)
        print(f"  데이터 타입: {result_for_saving.dtype}")
        print(f"  메모리 용량: {mb_size:.2f} MB")

        # subject_id 추출 (파일명에서 people 매칭)
        subject_id = 'unknown'
        for p in people:
            if p in raw_fname:
                subject_id = p
                break

        with h5py.File(final_dir, 'w') as f:
            dset = f.create_dataset(
                'nir_uv_data', data=result_for_saving,
                compression='gzip', compression_opts=4
            )
            dset.attrs['subject_id'] = subject_id
            dset.attrs['fps'] = 30

            mask_dset = f.create_dataset(
                'valid_mask', data=valid_mask.cpu().numpy(),
                compression='gzip', compression_opts=4
            )
            mask_dset.attrs['fail_rate'] = landmark_fail_rate.item()

        print(f"  ✅ 저장완료: {final_dir}")

        del frames_xy, frames_clahe, landmarks_xyz, frames_uv, frames_uv_masked, frames_uv_masked_final, result_for_saving
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ {raw_fname} 처리 실패")
        print(f"에러: {e}")
        traceback.print_exc()