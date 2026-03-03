import cv2
import mediapipe as mp
import os
import numpy as np

# 1. 설정 및 그리기 도구 준비
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 경로 설정 (사용자 환경에 맞게 수정됨)
base_dir = '/home/son/orientation-uv-rppg/data/MR-NIRP/NIR'
file_name = 'Frame00002.pgm'
image_path = os.path.join(base_dir, file_name)
output_path = os.path.join('/home/son/orientation-uv-rppg/data/test_pics', 'resultlandmark_nir.jpg')

# 2. 이미지 로드 및 전처리
# PGM 파일을 그레이스케일로 읽습니다.
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Error: 이미지를 찾을 수 없습니다. 경로를 확인하세요: {image_path}")
    exit()

# MediaPipe는 RGB 이미지를 입력으로 받으므로 변환합니다.
# NIR 이미지는 흑백이지만 3채널로 만들어줘야 합니다.
image_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

# 시각화 결과를 그릴 복사본 생성
annotated_image = image_rgb.copy()

# 3. MediaPipe Face Mesh 실행
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,  # 눈동자 주변 등 더 정밀한 랜드마크 포함
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=True  # 정지 이미지이므로 True로 설정
) as face_mesh:
    
    # 이미 NumPy 배열이므로 .numpy() 없이 바로 넘깁니다.
    results = face_mesh.process(image_rgb)

    # 4. 결과 시각화
    if results.multi_face_landmarks:
        print(f"랜드마크 검출 성공! 얼굴 개수: {len(results.multi_face_landmarks)}")
        for face_landmarks in results.multi_face_landmarks:
            # 제공된 그리기 도구로 랜드마크와 연결선을 그립니다.
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            

    else:
        print("얼굴 랜드마크를 찾지 못했습니다. NIR 이미지의 대비를 확인해보세요.")

# 5. 결과 이미지 저장
# 보기 편하게 RGB에서 BGR로 다시 변환해서 저장합니다.
cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
print(f"결과 이미지가 저장되었습니다: {output_path}")