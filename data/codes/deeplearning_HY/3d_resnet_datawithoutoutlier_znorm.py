import os
# torch import 전에 반드시 실행
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import sys
from tqdm import tqdm 
from utils.models import *
import argparse, os
import random
import math
from torchvision import transforms
from torch import nn
from utils.utils_sig import *
from utils.Loss import *
from utils.train import *
from utils.dataloader_onesubject import *


data_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_normalizeX'

all_files = sorted(glob.glob(os.path.join(data_dir, 'final_dataset_*.h5')))
all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('.h5') ]

train_files = all_files[:1]
valid_files = all_files[:1]
test_files = all_files[:1]
#valid_files = all_files[12:14]
#test_files = all_files[14:]

train_dataset = rPPGDataset(train_files, mode='train', crop_len=300)
valid_dataset = rPPGDataset(valid_files, mode='val', crop_len=300)
test_dataset = rPPGDataset(test_files, mode='test', crop_len=300)
print(f"train: {len(train_dataset)}샘플")
print(f"val:   {len(valid_dataset)}샘플")   # N_seg * 0.2 * 3구간
print(f"test:  {len(test_dataset)}샘플")

batch_size = 4

#train_loader = DataLoader(train_dataset, batch_size=batch_size , shuffle=True, collate_fn=collate_fn, num_workers=4)
#valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
#test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=batch_size , shuffle=True,  num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,  num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#for inputs, targets,pos_pad, pad_mask  in train_loader:
for inputs, targets  in train_loader:
    print("\n[DataLoader Test]")
    print(f"Input Video Batch Shape: {inputs.shape}")  # 예상: [4, 1, 900, 64, 64]
    print(f"Target GT Batch Shape: {targets.shape}")   # 예상: [4, 900]
    break

'''

# 배치 1개
tokens_pad, gt_batch, pos_pad, pad_mask = next(iter(train_loader))

gt = gt_batch[0].numpy()      # (T,)
fs = 30

# ── 1. GT band energy ──
b, a = butter(2, [0.75/(fs/2), 2.5/(fs/2)], btype='band')
gt_filtered = filtfilt(b, a, gt)
gt_ratio = np.var(gt_filtered) / (np.var(gt) + 1e-8)
print(f'GT band energy ratio: {gt_ratio:.4f}')

# ── 2. token band energy ──
tokens  = tokens_pad[0]          # (N, T)
valid   = ~pad_mask[0]           # (N,)
valid_ts = tokens[valid].numpy() # (N_valid, T)

ratios = []
for i in range(min(20, len(valid_ts))):
    ts = valid_ts[i]
    ts_f = filtfilt(b, a, ts)
    ratios.append(np.var(ts_f) / (np.var(ts) + 1e-8))

print(f'Token band energy ratios: {[f"{r:.4f}" for r in ratios]}')
print(f'Token band energy mean  : {np.mean(ratios):.4f}')

'''

# 설정 부분은 그대로 유지
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 
#model = rPPGTransformer().to(device) 
#criterion = RhythmFormer_Loss().to(device)
criterion = Neg_Pearson().to(device)
lr_ = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)
num_epochs = 50

best_val_loss = float('inf')
#main_pbar = tqdm(range(num_epochs), desc="Total Epochs")

train_loss_list = []
valid_loss_list = []
train_mae_list, valid_mae_list = [], []

test_loss_list, test_mae_list = [], []

for epoch in range(num_epochs):
    train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_mae, val_denom = validate(model, valid_loader, criterion, device)
    test_loss, test_mae,_ = validate(model, test_loader, criterion, device)

    #val_loss, val_mae,val_time,val_CE,val_HR = validate(model, valid_loader, criterion, device)
    #test_loss, test_mae,test_time,test_CE,test_HR = validate(model, test_loader, criterion, device)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)
    train_mae_list.append(train_mae)
    valid_mae_list.append(val_mae)

    #test_loss_list.append(test_loss)
    test_mae_list.append(test_mae)

    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] "
               f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} BPM | "
               f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f} BPM | "
               f"Test MAE: {test_mae:.2f} BPM, Val numer: {val_denom:.4f}") 
                   
                   
                   #f"Val PCC: {val_time:.4f}, Val CE: {val_CE:.2f}, Val HR: {val_HR:.2f} | "
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'3dres_data_wo_ol_lr:{str(lr_)}_epoch:{str(epoch)}.pth')
