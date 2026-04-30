import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils.utils_sig import *
from utils.Loss import *
from utils.train import *
from utils.dataloader import *
from utils.models import *

print('LOSO / driving_cropped / 300frame / NCorshift/ withoutregis/ physformer model')

data_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/driving_cropped_dl_data'
save_root = '/home/son/orientation-uv-rppg/data/parameter_save/nirpdriving/physformer/LOSO_driving_cropped_ncorshift'
shift_root = '/home/son/orientation-uv-rppg/data/shift_list_save/nirpdriving/physformer/LOSO_driving_cropped_ncorshift'

all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')])
n_subjects = len(all_files)
print(f"Total subjects: {n_subjects}")
for i, f in enumerate(all_files):
    print(f"  [{i}] {os.path.basename(f)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = rPPG_3D_ResNet().to(device) 
base_model = ViT_ST_ST_Compact3_TDC_gra_sharp(
    image_size=(300, 64, 64), 
    patches=(4, 4, 4), # 중요: 공간 해상도를 4x4로 맞추기 위한 셋업
    in_channels=1, 
    theta=0.2 # CDC_T 활성화
)

class PhysFormerWrapper(nn.Module):
    def __init__(self, base_model, gra_sharp=2.0):
        super().__init__()
        self.model = base_model
        self.gra_sharp = gra_sharp # 논문 디폴트 2.0

    def forward(self, x):
        # 1. 원본 모델에 필수 인자인 gra_sharp를 함께 넘겨줌
        # 2. 반환되는 (rPPG, Score1, Score2, Score3) 중 첫 번째 rPPG만 가져옴
        rPPG, _, _, _ = self.model(x, self.gra_sharp)
        return rPPG




lr_ = 1e-5
num_epochs = 40
batch_size = 2
fs = 30

fold_results = []

for fold_idx in range(n_subjects):
    test_files  = [all_files[fold_idx]]
    train_files = [all_files[i] for i in range(n_subjects) if i != fold_idx]

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: test={os.path.basename(test_files[0])}, train: {len(train_files)}")

    fold_save_dir = os.path.join(save_root, f'fold_{fold_idx}')
    fold_shift_dir = os.path.join(shift_root, f'fold_{fold_idx}')
    os.makedirs(fold_save_dir, exist_ok=True)
    os.makedirs(fold_shift_dir, exist_ok=True)

    train_loader = DataLoader(rPPGDataset(train_files), batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(rPPGDataset(test_files),  batch_size=batch_size, shuffle=False, num_workers=4)
    base_model = ViT_ST_ST_Compact3_TDC_gra_sharp(
    image_size=(300, 64, 64), 
    patches=(4, 4, 4), # 중요: 공간 해상도를 4x4로 맞추기 위한 셋업
    in_channels=1, 
    theta=0.2 # CDC_T 활성화
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysFormerWrapper(base_model).to(device)
    criterion = NCorshift()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)

    best_test_loss = float('inf')
    best_test_mae = None

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1, fs)
        test_metrics  = validate(model, test_loader, criterion, device, epoch+1, fs)

        tqdm.write(
            f"Fold {fold_idx} Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f} | "
            f"Test Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f}"
        )

        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            best_test_mae = test_metrics['mae']
            torch.save(model.state_dict(), os.path.join(fold_save_dir, 'best_model.pth'))

            if 'shift_list' in test_metrics:
                np.save(os.path.join(fold_shift_dir, 'best_test_shifts.npy'), test_metrics['shift_list'])

    fold_results.append({
        'fold': fold_idx,
        'test_subject': os.path.basename(test_files[0]),
        'best_test_loss': best_test_loss,
        'best_test_mae': best_test_mae
    })

# 최종 요약
print(f"\n{'='*60}")
print("LOSO Results Summary")
print(f"{'='*60}")
losses, maes = [], []
for r in fold_results:
    print(f"Fold {r['fold']:2d} | {r['test_subject']:40s} | Loss: {r['best_test_loss']:.4f} | MAE: {r['best_test_mae']:.2f}")
    losses.append(r['best_test_loss'])
    maes.append(r['best_test_mae'])

print(f"{'='*60}")
print(f"Mean Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
print(f"Mean MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f}")