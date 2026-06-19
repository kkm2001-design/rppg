import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import sys
from tqdm import tqdm 
import argparse, os
import random
import math
from torchvision import transforms
from torch import nn
from utils.utils_sig import *
from utils.Loss import *
from utils.train import *
from utils.dataloader import *
from utils.models import *
from utils.train_without_ce import *
import gc

##  2. 13번 subject를 뺏을떄 mae가 적절하게 나오는지 13번의 히스토그램상 혼자 이상해서 예를 빼는게맞는지 혹은 train에 넣기

print('physformer/ klloss')

data_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/small_motion_cropped_dl_data'
parameter_save_dir = '/home/son/orientation-uv-rppg/data/parameter_save/small_motion_cropped_dl_data/physformer/ncor+div'

os.makedirs(parameter_save_dir, exist_ok=True)


all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('garage_small_motion_940_dldata.h5')]

n_subjects = len(all_files)
print(f"Total subjects: {n_subjects}")
for i, f in enumerate(all_files):
    print(f"  [{i}] {os.path.basename(f)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhysFormerWrapper(nn.Module):
    def __init__(self, base_model, gra_sharp=2.0):
        super().__init__()
        self.model = base_model
        self.gra_sharp = gra_sharp  # 논문 디폴트 2.0

    def forward(self, x):
        # 1. 원본 모델에 필수 인자인 gra_sharp를 함께 넘겨줌
        # 2. 반환되는 (rPPG, Score1, Score2, Score3) 중 첫 번째 rPPG만 가져옴
        rPPG, _, _, _ = self.model(x, self.gra_sharp)
        return rPPG


lr_ = 1e-4
num_epochs = 80
batch_size = 2
fs = 30

fold_results = []

for fold_idx in range(n_subjects):

    if fold_idx > 0:
        del model, base_model, optimizer, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # ★ fold마다 동일 seed로 재현성 확보 (필요시 42 + fold_idx로 바꿔도 됨)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    test_files  = [all_files[fold_idx]]
    train_files = [all_files[i] for i in range(n_subjects) if i != fold_idx]

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: test={os.path.basename(test_files[0])}, train: {len(train_files)}")

    fold_save_dir = os.path.join(parameter_save_dir, f'fold_{fold_idx}')
    os.makedirs(fold_save_dir, exist_ok=True)

    train_loader = DataLoader(rPPGDataset(train_files), batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(rPPGDataset(test_files),  batch_size=batch_size, shuffle=False, num_workers=4)

    # ★ 매 fold마다 새 모델 인스턴스 생성 (가중치 초기화 보장)
    base_model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(300, 64, 64),
        patches=(4, 4, 4),  # 중요: 공간 해상도를 4x4로 맞추기 위한 셋업
        in_channels=1,
        theta=0.2  # CDC_T 활성화
    )
    model = PhysFormerWrapper(base_model).to(device)

    # sanity check: fold마다 init weight가 동일한지 확인 (seed 고정 시)
    with torch.no_grad():
        first_w = next(model.parameters()).data.flatten()[0].item()
    print(f"  [sanity] fold {fold_idx} first param = {first_w:.6f}")

    criterion = physformerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)

    best_test_loss = float('inf')
    best_test_mae = float('inf')
    mae_at_best_test_loss = 0
    best_mae_ckpt_path = None  # ★ 직전 best mae checkpoint 추적 (디스크 누적 방지)

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1, fs)
        test_metrics  = validate(model, test_loader, criterion, device, epoch+1, fs)

        if isinstance(criterion, physformerLoss):
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} NCOR:{train_metrics['ncor']:.4f}, kl:{train_metrics['kl']:.4f}, MAE: {train_metrics['mae']:.2f} | "
                f"Val Loss: {test_metrics['loss']:.4f} NCOR:{test_metrics['ncor']:.4f}, kl:{test_metrics['kl']:.4f}, MAE: {test_metrics['mae']:.2f} | "
            )
        else:
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f} | "
                f"Val Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f} | "
            )

        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            mae_at_best_test_loss = test_metrics['mae']

        if test_metrics['mae'] < best_test_mae:
            best_test_mae = test_metrics['mae']

            # ★ atomic save + 이전 best 삭제 (디스크 누적/중단 시 손상 방지)
            new_path = os.path.join(fold_save_dir, f'best_model{epoch}epoch.pth')
            tmp_path = new_path + '.tmp'
            torch.save(model.state_dict(), tmp_path)
            os.replace(tmp_path, new_path)

            if best_mae_ckpt_path is not None and os.path.exists(best_mae_ckpt_path):
                os.remove(best_mae_ckpt_path)
            best_mae_ckpt_path = new_path

    fold_results.append({
        'fold': fold_idx,
        'test_subject': os.path.basename(test_files[0]),
        'mabt': mae_at_best_test_loss,
        'best_test_loss': best_test_loss,
        'best_test_mae': best_test_mae
    })

# 최종 요약
print(f"\n{'='*60}")
print("LOSO Results Summary")
print(f"{'='*60}")
losses, maes, bmaes = [], [], []
for r in fold_results:
    print(f"Fold {r['fold']:2d} | {r['test_subject']:40s} | Loss: {r['best_test_loss']:.4f} | mae_at_best_loss : {r['mabt']:.2f} | bestMAE: {r['best_test_mae']:.2f}")
    losses.append(r['best_test_loss'])
    maes.append(r['mabt'])
    bmaes.append(r['best_test_mae'])

print(f"{'='*60}")
print(f"Mean Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
print(f"Mean MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f}")
print(f"Mean best_MAE:  {np.mean(bmaes):.2f} ± {np.std(bmaes):.2f}")