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
from utils_km import MediaPipeLandmarkDetector, GPUPiecewiseAffineTransform, BatchUVAngleMasker


############################################################################################################

############################### 비디오파일 만들고 그로부터 uv평면이동 전처리수행하는파일 #################################

############################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#==============================================================================================
                                ######## parameters #########
#==============================================================================================
thresh_angle = 45.0 




# ==============================================================================================
# 1. 경로 설정 (필요시 이후에 수정)
# ==============================================================================================
base_dir_raw_video = '/nasdata2/2yhy/CNA_ours_IRB/raw_frame_files/CNA_member_raw_frames' 
save_folder_raw = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/raw_video'
preprocess_result_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/preprocess_result'

people = ['A001_20250609','A002_20250609','A004_20250610','A005_20250611','A006_20250611','A007_20250612','A008_20250612','B001_20250623',
        'B002_20250623','B003_20250624','B004_20250624','B005_20250625','B008_20250627','C001_20250701','C002_20250703','C003_20250702']
rere = [8, 11, 15]

clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))

for i in range(len(people)):

    try:
        print("start the process....")

        # --------------------------------------------------------------------------------------
        # [make_video_2.py] 1. png 파일들 합쳐서 h5 영상파일 만들기
        # --------------------------------------------------------------------------------------
        ##########subject 설정##############
        subject_num = i ### subject 설정 ####
        vid_dir = os.path.join(base_dir_raw_video, people[subject_num])
        print(f"\n 데이터 경로 : {vid_dir}")

        ##########저장 파일 이름 설정##########
        output_name = f'raw_video_[{people[subject_num]}].h5'
        save_path = os.path.join(save_folder_raw, output_name)


        print(f"총 프레임 수 : {len(os.listdir(vid_dir))} ")

        #h5 자료형으로 저장 
        r_start, r_end, c_start, c_end = 0, 220, 100, 450

        if i in rere :
            r_start, r_end, c_start, c_end = 0, 300, 100, 550 ## crop 좌표 설정

        h, w = (r_end - r_start) , (c_end - c_start)


        files = sorted([f for f in os.listdir(vid_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])

        num_frames = len(files) # 프레임수 확인

        print(f"\n영상 생성 시작 {save_path} 경로에 생성 중 . . . ")

        with h5py.File(save_path, 'w') as f :
            
            dset = f.create_dataset('video', (num_frames, h, w), dtype = 'uint8', chunks=(1, h, w), compression = 'gzip', compression_opts=4)

            for f_i, f_name in enumerate(tqdm(files)) : 
                img = cv2.imread(os.path.join(vid_dir, f_name), cv2.IMREAD_UNCHANGED)
                if img is None : continue

                img_c = img[r_start : r_end , c_start : c_end]
                #img_rgb = cv2.cvtColor(img_c, cv2.COLOR_GRAY2RGB)

                dset[f_i] = img_c

        print(f"\n 저장완료 ! {num_frames}장의 프레임이 결합되었습니다.") 
        print(f"\n 저장경로 : {save_path}, subject-num : {people[subject_num]}") 

        # --------------------------------------------------------------------------------------
        # [extract_rppg.ipynb] 2. 데이터 전처리 로직 (바로 이어서 수행)
        # --------------------------------------------------------------------------------------
        print('-'*20,'[step1] 데이터 로드','-'*20)
        
        frames_xy = []
        frames_clahe_list = []
        
        # 방금 위에서 만든 H5 파일을 그대로 읽어옵니다
        h_dir = save_path

        with h5py.File(h_dir, 'r') as f_r:

            if 'video' in f_r:
                video_shape = f_r['video'].shape
                print(f"Video Dataset Shape: {video_shape}")

            else:
                print("'video' key를 찾을 수 없습니다.")

            h5_v = f_r['video']

            print("Converting frames to RGB...")
            for frame in tqdm(h5_v, desc = 'RGB Conversion') :
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                frames_xy.append((frame_rgb))

            print("Converting frames to Clahe...")
            for frame in tqdm(h5_v, desc = 'Clahe Conversion') :
                frame_clahe = clahe.apply(frame)
                frame_clahe_rgb = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2RGB)
                frames_clahe_list.append(frame_clahe_rgb)

        frames_xy = np.array(frames_xy)
        frames_clahe = np.array(frames_clahe_list)
        del frames_clahe_list

        print(f" 프레임 총 개수 : {len(frames_xy)}, 프레임 shape : {frames_xy[0].shape}")

        num_frames_original = len(frames_xy)
        print('-'*20,'[step1] 데이터 로드 완료','-'*20)

        # ====================================================
        print('-'*20,'[step2] 랜드마크 추출 시작','-'*20)

        detector = MediaPipeLandmarkDetector(min_detection_confidence=0.45, min_tracking_confidence=0.45)
        landmarks_xyz, valid_mask = detector(frames_clahe)
        landmark_fail_rate = ((num_frames_original - valid_mask.sum()) / num_frames_original)* 100

        print(f' 추출결과 shape : {landmarks_xyz.shape}') 
        print(f" 랜드마크 추출 성공 비율 :  {100 - landmark_fail_rate} %")

        print('-'*20,'✅[step2] 랜드마크 추출 완료','-'*20)

        # ====================================================
        print('-'*20,'[step3] UV domain transformation  시작','-'*20)

        valid_mask = valid_mask.detach().cpu()
        valid_indices = np.where(valid_mask.numpy())[0] 
        num_valid_frames = len(valid_indices)

        landmarks_xy = landmarks_xyz[:, :, :2] 
        landmarks_xy_valid = landmarks_xy[valid_mask] 
        landmarks_xyz_valid = landmarks_xyz[valid_mask] 

        num_frames_uv = landmarks_xy_valid.shape[0] # <== 수정: 변수 꼬임 방지 (num_frames -> num_frames_uv)
        batch_size = 64  
        frames_uv_list = []

        warper = GPUPiecewiseAffineTransform(detector.landmarks_uv, device=device)

        print(f"Starting Piece-wise Affine Transform... Total: {num_frames_uv} frames")

        with torch.no_grad():
            for j in tqdm(range(0, num_valid_frames, batch_size), desc="Warping Batches"): # <== 수정: 변수 꼬임 방지 (i -> j)

                current_batch_indices = valid_indices[j : j + batch_size]
                b_frames = frames_xy[current_batch_indices] 
                b_frames_gpu = torch.from_numpy(b_frames).to(device)

                b_landmarks_gpu = landmarks_xy_valid[j : j + batch_size]

                b_uv_gpu = warper(b_frames_gpu, b_landmarks_gpu)
                frames_uv_list.append(b_uv_gpu.cpu())

                if j % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()

        frames_uv = torch.cat(frames_uv_list, dim=0)

        print(f"최종 결과 Shape: {frames_uv.shape}") 
        print ('-'*20,'✅[step3] UV domain transformation' ,'-'*20)

        # ====================================================
        print('-'*20,'[step4] UV domain masking based on angle 시작','-'*20)

        masker = BatchUVAngleMasker(
            detector.mesh_faces,
            detector.landmarks_uv,
            device=device
        )

        num_frames_masked = frames_uv.shape[0] # <== 수정: 변수 꼬임 방지 (num_frames -> num_frames_masked)
        batch_size = 64  
        frames_masked_list = []

        print(f"Starting Surface Orientation Masking... Total: {num_frames_masked} frames")

        with torch.no_grad():  
            for j in tqdm(range(0, num_frames_masked, batch_size), desc="Masking Batches"): # <== 수정: 변수 꼬임 방지 (i -> j)

                b_frames_uv = frames_uv[j : j + batch_size].to(device)
                b_landmarks_xyz = landmarks_xyz_valid[j : j + batch_size]
                
                if not b_landmarks_xyz.is_cuda:
                    b_landmarks_xyz = b_landmarks_xyz.to(device)

                b_masked_gpu = masker(b_frames_uv, b_landmarks_xyz, degree_threshold= thresh_angle)
                frames_masked_list.append(b_masked_gpu.cpu())

                del b_frames_uv, b_landmarks_xyz, b_masked_gpu

                if j % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()

        frames_uv_masked = torch.cat(frames_masked_list, dim=0)

        print(f"Final Masked Shape: {frames_uv_masked.shape}") 
        print ('-'*20,'✅[step4] UV domain masking based on angle 완료' ,'-'*20)

        # ====================================================
        print('-'*20,'[step5] 결과저장 시작','-'*20)

        frames_uv_masked = frames_uv_masked.detach().cpu().numpy()
        frames_uv_masked_final = np.zeros((num_frames_original , 64, 64, 3), dtype=np.float32)
        frames_uv_masked_final[valid_mask] = frames_uv_masked

        print(f" 최종결과 : {frames_uv_masked_final.shape}")

        ## save final data
        result_for_saving = frames_uv_masked_final[:, :, :, 0]

        if result_for_saving.dtype == np.float64 :
            result_for_saving = result_for_saving.astype(np.float32)
            print("자료형 변환완료 : float64 -> float32 ")

        
        fil_nam = output_name.replace('raw_video', 'registration_video')
        finall_dir = os.path.join(preprocess_result_dir, fil_nam)

        byte_size = result_for_saving.nbytes
        mb_size = byte_size / (1024 ** 2)
        gb_size = byte_size / (1024 ** 3)

        print(f"데이터 타입: {result_for_saving.dtype}")
        print(f"전체 요소 개수: {result_for_saving.size:,} 개")
        print(f"메모리 용량: {mb_size:.2f} MB ({gb_size:.2f} GB)")

        with h5py.File( finall_dir, 'w') as f :
            dset = f.create_dataset(
                'nir_uv_data',
                data = result_for_saving,
                compression = 'gzip',
                compression_opts = 4
            )
            dset.attrs['subject_id'] = people[subject_num]
            dset.attrs['fps'] = 30

            mask_dset = f.create_dataset(
                'valid_mask',
                data = valid_mask.cpu().numpy(), 
                compression = 'gzip',
                compression_opts = 4
            )
            mask_dset.attrs['fail_rate'] = landmark_fail_rate.item()

            print(f"✅ [전체 process 완료] {finall_dir} 경로에 저장 완료!")

            del frames_xy, frames_clahe, landmarks_xyz, frames_uv, frames_uv_masked_final, result_for_saving
            torch.cuda.empty_cache()

    except Exception as e :
        print("\n\n" + "="*50)
        print("❌ [치명적 오류 발생] 프로그램이 중단되었습니다.")
        print("="*50)
        print(f"에러 메시지: {e}")
        print("-" * 30)
        print("상세 에러 위치 (Traceback):")
        traceback.print_exc() 
        print("="*50)