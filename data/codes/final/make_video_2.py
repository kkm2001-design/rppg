import cv2
import os
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

############################################################################################################

#################################### png 파일들합쳐서 h5 영상파일 만드는 파일 ######################################

############################################################################################################

# 1. 경로 설정 (절대 경로 사용)
base_dir_raw_video = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/videos/' 
people = sorted([f for f in os.listdir(base_dir_raw_video)])
save_folder_raw = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/raw_video' 
rere = [8, 11, 15]
#print(people)

for i in rere :

    ##########subject 설정##############
    subject_num = i ### subject 설정 ####
    vid_dir = os.path.join(base_dir_raw_video, people[subject_num],'NIR')
    print(f"\n 데이터 경로 : {vid_dir}")

    ##########저장 파일 이름 설정##########
    output_name = f'nirp_cna_video_[{people[subject_num]}].h5'
    save_path = os.path.join(save_folder_raw, output_name)


    print(f"총 프레임 수 : {len(os.listdir(vid_dir))} ")

    #h5 자료형으로 저장 

    r_start, r_end, c_start, c_end = 0, 300, 100, 550 ## crop 좌표 설정
    h, w = (r_end - r_start) , (c_end - c_start)

    files = sorted([f for f in os.listdir(vid_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])

    num_frames = len(files) # 프레임수 확인

    print(f"\n영상 생성 시작 {save_path} 경로에 생성 중 . . . ")

    with h5py.File(save_path, 'w') as f :
        
        dset = f.create_dataset('video', (num_frames, h, w), dtype = 'uint8', chunks=(1, h, w), compression = 'gzip',compression_opts=4)

        for i, f_name in enumerate(tqdm(files)) : 
            img = cv2.imread( os.path.join(vid_dir, f_name), cv2.IMREAD_UNCHANGED)
            if img is None : continue

            img_c = img[r_start : r_end , c_start : c_end]
            #img_rgb = cv2.cvtColor(img_c, cv2.COLOR_GRAY2RGB)

            dset[i] = img_c

    print(f"\n 저장완료 ! {num_frames}장의 프레임이 결합되었습니다.") 
    print(f"\n 저장경로 : {save_path}, subject-num : {people[subject_num]}") 
            