import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
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


print('3dphysnettest train_files = all_files[:12] valid_files = all_files[12:14] test_files = all_files[:]')

data_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier'
all_files = sorted(glob.glob(os.path.join(data_dir, 'final_dataset_*.h5')))
all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('.h5') ]

train_files = all_files[:12]
valid_files = all_files[12:14]

train_dataset = rPPGDataset(train_files)
valid_dataset = rPPGDataset(valid_files)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True, num_workers =4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for inputs, targets in train_loader:
    print("\n[DataLoader Test]")
    print(f"Input Video Batch Shape: {inputs.shape}")  # 예상: [4, 1, 900, 64, 64]
    print(f"Target GT Batch Shape: {targets.shape}")   # 예상: [4, 900]
    break


# 설정 부분은 그대로 유지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = rPPG_3D_ResNet().to(device) 
model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 
criterion = Neg_Pearson().to(device)
lr_ = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)
num_epochs = 200

best_val_loss = float('inf')
main_pbar = tqdm(range(num_epochs), desc="Total Epochs")

train_loss_list = []
valid_loss_list = []
train_mae_list, valid_mae_list = [], []


for epoch in main_pbar:
    train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_mae = validate(model, valid_loader, criterion, device)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)
    train_mae_list.append(train_mae)
    valid_mae_list.append(val_mae)

    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] "
               f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} BPM | "
               f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f} BPM")
    
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     #torch.save(model.state_dict(), f'3dres_data_wo_ol_lr:{str(lr_)}_epoch:{str(epoch)}.pth')




