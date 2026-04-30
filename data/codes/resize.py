import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np
from pathlib import Path
import cv2

data_dir = Path("/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/driving_cropped_dl_data")

for h5_path in sorted(data_dir.glob("*.h5")):
    print(f"Processing: {h5_path.name}")
    
    with h5py.File(h5_path, 'r+') as f:
        if 'video' not in f:
            print(f"  'video' key not found, skipping.")
            continue
        
        video = f['video'][:]  # (8, 300, 128, 128)
        print(f"  Original shape: {video.shape}, dtype: {video.dtype}")
        
        # (8, 300, 128, 128) -> (8, 300, 64, 64)
        resized = np.zeros((video.shape[0], video.shape[1], 64, 64), dtype=video.dtype)
        
        for i in range(video.shape[0]):
            for t in range(video.shape[1]):
                resized[i, t] = cv2.resize(video[i, t], (64, 64), interpolation=cv2.INTER_AREA)
        
        # 기존 dataset 삭제 후 재생성
        del f['video']
        f.create_dataset('video', data=resized, compression='gzip')
        print(f"  Resized shape: {resized.shape}")

print("Done.")