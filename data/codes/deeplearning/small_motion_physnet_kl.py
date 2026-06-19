import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
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


print('physnet/ klloss')

data_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/small_motion_cropped_dl_data'
parameter_save_dir = '/home/son/orientation-uv-rppg/data/parameter_save/small_motion_cropped_dl_data/physnet/kl'

os.makedirs(parameter_save_dir, exist_ok= True)


all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('garage_small_motion_940_dldata.h5') ]

n_subjects = len(all_files)
print(f"Total subjects: {n_subjects}")
for i, f in enumerate(all_files):
    print(f"  [{i}] {os.path.basename(f)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 


lr_ = 1e-4
num_epochs = 80
batch_size = 24
fs = 30

fold_results = []

for fold_idx in range(n_subjects):

    if fold_idx > 0:
        del model, optimizer, train_loader, test_loader
        import gc; gc.collect()
        torch.cuda.empty_cache()

    test_files  = [all_files[fold_idx]]
    train_files = [all_files[i] for i in range(n_subjects) if i != fold_idx]

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: test={os.path.basename(test_files[0])}, train: {len(train_files)}")

    fold_save_dir = os.path.join(parameter_save_dir, f'fold_{fold_idx}')

    os.makedirs(fold_save_dir, exist_ok=True)
  

    train_loader = DataLoader(rPPGDataset(train_files), batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(rPPGDataset(test_files),  batch_size=batch_size, shuffle=False, num_workers=4)

    model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 

    criterion = physformerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)

    best_test_loss = float('inf')
    best_test_mae = float('inf')
    mae_at_best_test_loss = 0


    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1, fs)
        test_metrics  = validate(model, test_loader, criterion, device, epoch+1, fs)

        if isinstance(criterion, physformerLoss) :
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} NCOR:{train_metrics['ncor']:.4f}, kl:{train_metrics['kl']:.4f}, MAE: {train_metrics['mae']:.2f} | " 
                f"Val Loss: {test_metrics['loss']:.4f} NCOR:{test_metrics['ncor']:.4f}, kl:{test_metrics['kl']:.4f}, MAE: {test_metrics['mae']:.2f} | "
            )

        else :
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f} | " 
                f"Val Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f} | "
        )

        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            mae_at_best_test_loss = test_metrics['mae']
   


        if test_metrics['mae'] < best_test_mae  :
            best_test_mae = test_metrics['mae']
            torch.save(model.state_dict(), os.path.join(fold_save_dir, f'best_model{epoch}epoch.pth'))


    fold_results.append({
        'fold': fold_idx,
        'test_subject': os.path.basename(test_files[0]),
        'mabt' : mae_at_best_test_loss,
        'best_test_loss': best_test_loss,
        'best_test_mae': best_test_mae
    })

# 최종 요약
print(f"\n{'='*60}")
print("LOSO Results Summary")
print(f"{'='*60}")
losses, maes, bmaes = [], [], []
for r in fold_results:
    print(f"Fold {r['fold']:2d} | {r['test_subject']:40s} | Loss: {r['best_test_loss']:.4f} | mae_at_best_loss : {r['mabt'] :.2f} | bestMAE: {r['best_test_mae']:.2f}")
    losses.append(r['best_test_loss'])
    maes.append(r['mabt'])
    bmaes.append(r['best_test_mae'])

print(f"{'='*60}")
print(f"Mean Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
print(f"Mean MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f}")
print(f"Mean best_MAE:  {np.mean(bmaes):.2f} ± {np.std(bmaes):.2f}")




