import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import matplotlib.pyplot as plt
import numpy as np
h5_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/MR-nirp_driving_motion_registration'

# 저장할 폴더 생성


files = sorted([f for f in os.listdir(h5_dir) if f.endswith('large_motion_975_uv.h5')])


for sub_id in range(len(files)) :

    file = files[sub_id]
    print(f"Target file: {files[sub_id]}")

    with h5py.File(os.path.join(h5_dir, file), 'r') as f:
        video = f['nir_uv_data'][:]

    #video = np.concatenate(video, axis = 0)

    print(f"Video shape: {video.shape}") # (T, h, w, c) 형태 예상
    print(f"Video dtype: {video.dtype}")

    T = video.shape[0] # 전체 프레임 수

    save_dir = f'/home/son/orientation-uv-rppg/data/pics/driving_pics_regis/{file}'
    os.makedirs(save_dir, exist_ok=True)

    # 0부터 전체 프레임(T)까지 100씩 건너뛰며 반복
    for start_idx in range(0, T, 100):
        end_idx = min(start_idx + 100, T) # 마지막 구간이 100장이 안 될 경우 처리
        chunk_size = end_idx - start_idx
        
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        axes = axes.flatten()
        
        for i in range(100):
            if i < chunk_size:
                im = video[start_idx + i]
                
                # 채널(c) 차원이 1인 경우 (T, h, w, 1) 에러 방지용
                if len(im.shape) == 3 and im.shape[-1] == 1:
                    im = im.squeeze(-1)
                    
                axes[i].imshow(im, cmap='gray')
                axes[i].set_title(f'frame: {start_idx + i}')
            
            # 프레임이 100개가 안 채워지거나, 더 깔끔하게 보기 위해 축(axis) 숨김
            axes[i].axis('off') 

        plt.tight_layout()
        
        # 100장 단위로 묶인 이미지를 파일로 저장 (예: frames_0_to_99.png)
        save_path = os.path.join(save_dir, f'frames_{start_idx}_to_{end_idx-1}.png')
        plt.savefig(save_path)
        
        # 메모리 릭(누수) 방지를 위해 figure를 반드시 닫아줌
        plt.close(fig) 
        
        print(f"Saved: {save_path}")

print("모든 프레임 저장이 완료되었습니다!")