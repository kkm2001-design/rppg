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

# 1. 경로 설정
h5_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration'
base_save_dir = '/home/son/orientation-uv-rppg/data/pics/B008pics'
os.makedirs(base_save_dir, exist_ok=True)

files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
id = 12

print(f"[{files[id]}] 파일을 불러옵니다...")
with h5py.File(os.path.join(h5_dir, files[id]), 'r') as f:
    video = f['video'][:] 

print(f"로드된 비디오 Shape: {video.shape}") # 예상 출력: (76, 900, 300, 450)

num_samples = video.shape[0]  # 76
num_frames = video.shape[1]   # 900

# 3. 76개의 샘플을 순회하며 900장 모두 저장 (100장씩 9번)
for v_idx in range(num_samples):
    print(f"[{v_idx+1}/{num_samples}] 번째 샘플 (Index {v_idx}) 처리 중...")
    

    ss = 9 * v_idx
    frames_per_grid = 100
    num_grids = int(np.ceil(num_frames / frames_per_grid)) # 900/100 = 9개
    
    for g in range(num_grids):
        start_idx = g * frames_per_grid
        
        # 10x10 그리드 생성
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        axes = axes.flatten()
        
        for i in range(frames_per_grid):
            frame_idx = start_idx + i
            
            # 900장 범위를 넘어가지 않도록 체크
            if frame_idx < num_frames:
                # video shape이 (76, 900, 300, 450) 이므로
                # v_idx번째 샘플의 frame_idx번째 이미지 추출
                img = video[v_idx, frame_idx] 
                
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'F:{frame_idx}', fontsize=8)
            
            axes[i].axis('off') # 테두리/눈금 제거
        
        plt.tight_layout()
        
        # 저장 파일명: grid_0.png ~ grid_8.png
        save_name = f"grid_{g + ss }.png" 
        plt.savefig(os.path.join(base_save_dir, save_name), dpi=100)
        plt.close() # 메모리 누수 방지

print("🚀 76개 샘플의 모든 프레임 저장이 완료되었습니다!")