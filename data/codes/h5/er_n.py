h5_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data'
subject_lists = sorted( [f for f in os.listdir(h5_dir) if f.endswith('.h5') ])
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))

base_dirrr = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/videos/'
people11 = sorted([f for f in os.listdir(base_dirrr)])

print(people11)

for sub_n in range(2:16) : 
 
    try :
        print("strat the process....")

        # ====================================================
        # 1. 데이터 로드
        # ====================================================
        print('-'*20,'[step1] 데이터 로드','-'*20)
        
        frames_xy = []
        frames_clahe_list = []
        subject_number = sub_n ## 처리할 subject 번호
        h_dir = os.path.join(h5_dir,subject_lists[subject_number])

        with h5py.File(h_dir, 'r') as f_r:
            # 데이터셋 키(key) 확인 (파일 안에 어떤 데이터셋이 있는지 모를 때)
            print("Keys in H5 file:", list(f_r.keys()))

            # 'video' 데이터셋의 shape 확인
            if 'video' in f_r:
                video_shape = f_r['video'].shape
                print(f"Video Dataset Shape: {video_shape}")

                # 추가 정보 (자료형 확인)
                print(f"Video Dataset Dtype: {f_r['video'].dtype}")
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

        frames_xy_torch = torch.from_numpy(frames_xy)

        num_frames_original = len(frames_xy)
        print('-'*20,'[step1] 데이터 로드 완료','-'*20)
        # ====================================================
        # 2. 랜드마크 추출
        # ====================================================
        print('-'*20,'[step2] 랜드마크 추출 시작','-'*20)

        detector = MediaPipeLandmarkDetector(min_detection_confidence=0.45, min_tracking_confidence=0.45)
        landmarks_xyz, valid_mask = detector(frames_clahe)
        landmark_success_rate = ((num_frames_original - valid_mask.sum()) / num_frames_original)* 100



        print(f' 추출결과 shape : {landmarks_xyz.shape}') # >>> [T,V,3]
        print(f" 랜드마크 추출 실패 비율 :  {landmark_success_rate} %")

        print('-'*20,'✅[step2] 랜드마크 추출 완료','-'*20)
        # ====================================================
        # 3. UV domain transformation
        # ====================================================
        print('-'*20,'[step3] UV domain transformation  시작','-'*20)

        valid_mask = valid_mask.detach().cpu()
        valid_indices = np.where(valid_mask.cpu().numpy())[0] # 인덱스 숫자만 추출
        num_valid_frames = len(valid_indices)

        landmarks_xy = landmarks_xyz[:, :, :2] # 랜드마크 2차원화
        landmarks_xy_valid = landmarks_xy[valid_mask] # 랜드마크 추출한 프레임들로 제한

        landmarks_xyz_valid = landmarks_xyz[valid_mask] # 랜드마크 추출한 프레임들로 제한

        num_frames = landmarks_xy_valid.shape[0]
        batch_size = 64  # GPU 메모리 상황에 따라 16, 32 등으로 조절하세요.
        frames_uv_list = []

        warper = GPUPiecewiseAffineTransform(detector.landmarks_uv, device=device)

        # 3. 배치 루프 시작
        print(f"Starting Piece-wise Affine Transform... Total: {num_frames} frames")

        # torch.no_grad()를 사용하여 연산 중 불필요한 메모리 할당 방지
        with torch.no_grad():
            for i in tqdm(range(0, num_valid_frames, batch_size), desc="Warping Batches"):

                current_batch_indices = valid_indices[i : i + batch_size]
                b_frames = frames_xy[current_batch_indices] # 필요한 만큼만 가져옴
                b_frames_gpu = torch.from_numpy(b_frames).to(device)

                b_landmarks_gpu = landmarks_xy_valid[i : i + batch_size]

                # 워핑 수행
                # warper 내부에서 frames / 255.0 등 연산이 일어나므로 원본 데이터 보호
                b_uv_gpu = warper(b_frames_gpu, b_landmarks_gpu)

                # 결과물은 다시 CPU로 내려서 리스트에 보관 (GPU 메모리 즉시 비우기)
                frames_uv_list.append(b_uv_gpu.cpu())

                # 매 루프 끝에 명시적인 GPU 캐시 비우기 (선택 사항)
                if i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()

        # 4. 결과 통합
        frames_uv = torch.cat(frames_uv_list, dim=0)


        print(f"최종 결과 Shape: {frames_uv.shape}") # [T, 64, 64, 3]
        print ('-'*20,'✅[step3] UV domain transformation' ,'-'*20)
        # ====================================================
        # 4. UV domain masking based on angle
        # ====================================================
        print('-'*20,'[step4] UV domain masking based on angle 시작','-'*20)

        # 1. Masker 초기화 (한 번만 수행)
        # faces와 landmarks_uv는 용량이 작으므로 초기화 시 GPU에 올려도 괜찮습니다.
        masker = BatchUVAngleMasker(
            detector.mesh_faces,
            detector.landmarks_uv,
            device=device
        )

        # 2. 데이터 준비
        # frames_uv는 직전 단계 결과물이므로 CPU에 있을 것입니다.
        # landmarks_xyz_valid는 GPU에 있더라도 슬라이싱하면 GPU 메모리를 크게 먹지 않습니다.
        num_frames = frames_uv.shape[0]
        batch_size = 64  # 앞서 성공했던 batch_size와 동일하게 맞추는 것을 추천합니다.
        frames_masked_list = []

        print(f"Starting Surface Orientation Masking... Total: {num_frames} frames")

        # 3. 배치 루프 실행
        with torch.no_grad():  # 그래디언트 계산 방지 (메모리 절약 필수)
            for i in tqdm(range(0, num_frames, batch_size), desc="Masking Batches"):

                # --- A. 데이터 슬라이싱 및 GPU 이동 ---
                # 1) UV 프레임: CPU -> GPU
                b_frames_uv = frames_uv[i : i + batch_size].to(device)

                # 2) 3D 랜드마크: (이미 GPU라면 슬라이싱만, CPU라면 이동)
                # 만약 landmarks_xyz_valid가 GPU 텐서라면 .to(device)는 생략해도 됩니다.
                b_landmarks_xyz = landmarks_xyz_valid[i : i + batch_size]
                if not b_landmarks_xyz.is_cuda:
                    b_landmarks_xyz = b_landmarks_xyz.to(device)

                # --- B. 마스킹 수행 ---
                # 내부에서 Normal 계산 -> 각도 계산 -> 마스킹 적용
                b_masked_gpu = masker(b_frames_uv, b_landmarks_xyz, degree_threshold=60.0)

                # --- C. 결과 저장 및 메모리 정리 ---
                # 결과물을 즉시 CPU로 내립니다.
                frames_masked_list.append(b_masked_gpu.cpu())

                # 사용한 임시 변수 삭제 (확실한 메모리 해제)
                del b_frames_uv, b_landmarks_xyz, b_masked_gpu

                # 주기적인 캐시 비우기
                if i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()

        # 4. 결과 통합
        frames_uv_masked = torch.cat(frames_masked_list, dim=0)

        print(f"Final Masked Shape: {frames_uv_masked.shape}") # [T, 64, 64, C]
        print ('-'*20,'✅[step4] UV domain masking based on angle 완료' ,'-'*20)

        # ====================================================
        # 5. 결과저장
        # ====================================================
        print('-'*20,'[step5] 결과저장 시작','-'*20)

        frames_uv_masked = frames_uv_masked.detach().cpu().numpy()

        frames_uv_masked_final = np.zeros((num_frames_original , 64, 64, 3), dtype=np.float32)

        frames_uv_masked_final[valid_mask] = frames_uv_masked

        print(f" 최종결과 : {frames_uv_masked_final.shape}")

        ## save final data
        preprocess_result_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/preprocess_result'
        result_for_saving = frames_uv_masked_final[:, :, :, 0]

        if result_for_saving.dtype == np.float64 :
            result_for_saving = result_for_saving.astype(np.float32)
            print("자료형 변환완료 : float64 -> float32 ")


        fil_nam = f'{subject_lists[subject_number]}'

        finall_dir = os.path.join(preprocess_result_dir, fil_nam)

        byte_size = result_for_saving.nbytes

        # 2. MB, GB 단위로 변환
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
            dset.attrs['subject_id'] = subject_lists[subject_number]
            dset.attrs['fps'] = 30


            mask_dset = f.create_dataset(
                'valid_mask',
                data = valid_mask.cpu().numpy(), # 가지고 계신 boolean 리스트/배열
                compression = 'gzip',
                compression_opts = 4
            )
            mask_dset.attrs['fail_rate'] = landmark_success_rate.item()

            print(f"✅ [전체 process 완료] {finall_dir} 경로에 저장 완료!")

    except Exception as e :

        print("\n\n" + "="*50)
        print("❌ [치명적 오류 발생] 프로그램이 중단되었습니다.")
        print("="*50)
        print(f"에러 메시지: {e}")
        print("-" * 30)
        print("상세 에러 위치 (Traceback):")
        traceback.print_exc() # 어디서 에러가 났는지 상세하게 줄 번호까지 출력해줍니다.
        print("="*50)


