import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import torch
import glob
from utils.dataloader import rPPGDataset
from utils.k_fold_test import *
from utils.models import *

model_select = PhysNet_padding_Encoder_Decoder_MAX

#model_select = rPPG_3D_ResNet   # 클래스 자체를 변수에 할당 (인스턴스 생성 아님)
#model_select                     # <class 'utils.models.rPPG_3D_ResNet'>
#model_select()                   # 이때 인스턴스 생성 (괄호 붙일 때)

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
DATA_DIR   = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/MR-nirp_driving_dl_data' ############################## dl_data 불러오는 경로
PIC_DIR    = f'/home/son/orientation-uv-rppg/data/pics/MR-nirp_driving/{model_select.__name__}/LOSO'       ################################# loss curve 사진 저장 경로 
PARAM_DIR  = f'/home/son/orientation-uv-rppg/data/codes/deeplearning/parameters_save/MR-nirp_driving/{model_select.__name__}/LOSO'  ################################ best parameter 저장 경로
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────────
# subject → file 매핑 구성
# ──────────────────────────────────────────────
subject_ids = [f'subject{i}' for i in range(1, 20)]
print(f'  MODEL : {model_select} / nirp_driving_kfold_test')
subject_file_map = {}
for sid in subject_ids:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f'{sid}_garage_still_940_uv_windowed.h5')))
    if files:
        subject_file_map[sid] = files

print("Subject-File map:")
for k, v in subject_file_map.items():
    print(f"  {k}: {[os.path.basename(f) for f in v]}")
print(f"총 subject 수: {len(subject_file_map)}")





results = run_kfold(
    model_fn         = lambda: model_select(frames=300),
    subject_file_map = subject_file_map,
    Dataset_cls      = rPPGDataset,
    norm_type        = 'fixed',
    device           = DEVICE,
    num_epochs       = 100,
    batch_size       = 8,
    lr               = 1e-4,
    pic_dir          = PIC_DIR,
    param_dir        = PARAM_DIR,
    tag              = f'{model_select.__name__}_fixed_1e-3_LOSO',
    k                = 4
)

# ──────────────────────────────────────────────
# 실험 2: PhysNet + zscore norm + LOSO  (비교)
# 실험 1 결과 확인 후 주석 해제
# ──────────────────────────────────────────────
# results = run_kfold(
#     model_fn         = lambda: PhysNet_padding_Encoder_Decoder_MAX(frames=300),
#     subject_file_map = subject_file_map,
#     Dataset_cls      = rPPGDataset,
#     norm_type        = 'zscore',
#     device           = DEVICE,
#     num_epochs       = 100,
#     batch_size       = 8,
#     lr               = 1e-4,
#     pic_dir          = PIC_DIR,
#     param_dir        = PARAM_DIR,
#     tag              = 'physnet_zscore_LOSO',
#     k                = None,
# )