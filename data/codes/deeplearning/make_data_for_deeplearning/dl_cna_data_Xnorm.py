import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np
from tqdm import tqdm

def robust_diff_scaling(data):
    """
    1차 차분 후 표준편차로 나누고 3-clipping 수행
    주행 데이터의 급격한 노이즈(Outlier)를 억제하는 데 최적화된 방식
    """
    # 1. 1차 차분 수행 (T축 기준)
    diff = np.diff(data, axis=0) # (299, 64, 64) 또는 (299,)
    
    # 2. 표준편차 계산 및 정규화
    std = np.std(diff)
    normalized_diff = diff / (std + 1e-8)
    diff = np.diff(data, axis=0) # (299, 64, 64) 또는 (299,)
    
    mean_val = np.mean(diff)
    std_val = np.std(diff)
    
    # 평균을 빼줌으로써 완벽한 Zero-centering 보장
    normalized_diff = (diff - mean_val) / (std_val + 1e-8)
    # 3. 3-clipping (평균 0 기준 +-3 범위 밖은 잘라냄)
    # 모델이 비정상적인 튀는 값에 과하게 반응하는 것을 방지
    clipped_diff = np.clip(normalized_diff, -3, 3)
    
    return clipped_diff

def process_robust_diff(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    for file_name in tqdm(h5_files, desc="Processing Robust Diff"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"robust_diff_{file_name}")

        with h5py.File(input_path, 'r') as f_in:
            video = f_in['video'][:]    # (N, 300, 64, 64)
            gt_ppg = f_in['gt_ppg'][:]  # (N, 300)
            hr = f_in['hr'][:]
            
            num_samples = video.shape[0]
            # 결과물 저장용 어레이 (프레임 수 299)
            processed_video = np.zeros((num_samples, 299, 64, 64), dtype=np.float32)
            processed_gt = np.zeros((num_samples, 299), dtype=np.float32)

            for i in range(num_samples):
                # 각 샘플(윈도우) 단위로 Robust Scaling 적용
                processed_video[i] = robust_diff_scaling(video[i])
                processed_gt[i] = robust_diff_scaling(gt_ppg[i])

            with h5py.File(output_path, 'w') as f_out:
                f_out.create_dataset('video', data=processed_video, compression="gzip")
                f_out.create_dataset('gt_ppg', data=processed_gt, compression="gzip")
                f_out.create_dataset('hr', data=hr.astype(np.float32))
                
                f_out.attrs['desc'] = "First-order diff + std-norm + 3-clipping applied"

# ===== 실행 =====
input_path = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped_len300'
output_path = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_diffnorm_len300'

process_robust_diff(input_path, output_path)