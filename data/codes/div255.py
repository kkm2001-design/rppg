import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np

data_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/driving_cropped_dl_data'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])

for fname in files:
    path = os.path.join(data_dir, fname)
    with h5py.File(path, 'a') as f:
        video = f['video'][:].astype(np.float32) / 255.0
        print(f"{fname} | shape: {video.shape} | min: {video.min():.4f}, max: {video.max():.4f}")
        del f['video']
        f.create_dataset('video', data=video, chunks=True, compression='gzip', compression_opts=4)
    print(f"  ✅ 완료")