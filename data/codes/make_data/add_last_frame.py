import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob
import numpy as np
import h5py
from tqdm import tqdm

# 경로 설정
input_dir = "/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_diffnorm_len300"
output_dir = "/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_diffnorm_len300_final"
os.makedirs(output_dir, exist_ok=True)

file_paths = glob.glob(os.path.join(input_dir, "*.h5"))
print(f"총 {len(file_paths)}개의 .h5 파일을 변환합니다. (video만 300프레임으로 패딩)")

for path in tqdm(file_paths):
    save_path = os.path.join(output_dir, os.path.basename(path))
    
    with h5py.File(path, 'r') as f_in, h5py.File(save_path, 'w') as f_out:
        
        # 1. video 처리: (349, 299, 64, 64) -> (349, 300, 64, 64)
        video = f_in['video'][()]
        last_frame = video[:, -1:, :, :]  # 마지막 프레임만 추출
        video_padded = np.concatenate([video, last_frame], axis=1) # 시간 축(axis=1)에 붙이기
        f_out.create_dataset('video', data=video_padded)
        
        # 2. gt_ppg 처리: 건드리지 않고 그대로 복사! (349, 299)
        gt_ppg = f_in['gt_ppg'][()]
        f_out.create_dataset('gt_ppg', data=gt_ppg)
        
        # 3. hr 처리: 건드리지 않고 그대로 복사! (349,)
        hr = f_in['hr'][()]
        f_out.create_dataset('hr', data=hr)

print("\n🎉 변환 완료! video만 300으로 늘리고 gt_ppg는 299 그대로 유지했습니다.")