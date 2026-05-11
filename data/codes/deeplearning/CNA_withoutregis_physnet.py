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
from utils.train_without_ce import *

##  2. 13번 subject를 뺏을떄 mae가 적절하게 나오는지 13번의 히스토그램상 혼자 이상해서 예를 빼는게맞는지 혹은 train에 넣기



data_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped'
parameter_save_dir = '/home/son/orientation-uv-rppg/data/parameter_save/900franewithoutregis_13out_and_znorm/bandpassbeforelosscompute_ncorshift'
shift_list_save_dir = '/home/son/orientation-uv-rppg/data/shift_list_save/CNA_data/without_regis_900frmae_bandpassbeforelosscompute_negcorshift'
os.makedirs(parameter_save_dir, exist_ok= True)
os.makedirs(shift_list_save_dir, exist_ok= True)

all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('.h5') ]

train_files = all_files[:12]
valid_files = all_files[13:14]  
test_files = all_files[14:]

train_dataset = rPPGDataset(train_files)
valid_dataset = rPPGDataset(valid_files)
test_dataset = rPPGDataset(test_files)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True, num_workers =4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# for inputs, targets, hr in train_loader:
#     print("\n[DataLoader Test]")
#     print(f"Input Video Batch Shape: {inputs.shape}")  # 예상: [8, 1, 900, 64, 64]
#     print(f"Target GT Batch Shape: {targets.shape}")   # 예상: [8, 900]
#     print(f"Target GT Hr value shape : {hr.shape}")
#     break


# 설정 부분은 그대로 유지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = rPPG_3D_ResNet().to(device) 
model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 

max_shift_sec = 1/2
shift_start_epoch = 30

criterion = physformerloss()
lr_ = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)
num_epochs = 100

print('3dphysnettest train_files = all_files[:12] valid_files = all_files[13:14]test_files = alt_files[14:]]')
print(f'900frame / gt subjectwise test / without registration/ 13subject out / bandpassbeforelosscompute_ncorshift/ shift_sec :{max_shift_sec}/ shift_start_epoch : {shift_start_epoch} ')


best_val_loss = float('inf')
main_pbar = tqdm(range(num_epochs), desc="Total Epochs")

train_loss_list = []
valid_loss_list = []
train_mae_list, valid_mae_list = [], []


for epoch in main_pbar:
    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
    val_metrics   = validate(model, valid_loader, criterion , device , epoch+1)
    test_metrics  = validate(model, test_loader, criterion, device , epoch+1)
    
    train_loss_list.append(train_metrics['loss'])
    valid_loss_list.append(val_metrics['loss'])
    train_mae_list.append(train_metrics['mae'])
    valid_mae_list.append(val_metrics['mae'])

    # 💡 HybridLoss 여부에 따라 출력 문자열 다르게 구성
    if isinstance(criterion, HybridLoss):
        tqdm.write(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} (NCOR:{train_metrics['ncor']:.4f}, MSE:{train_metrics['mse']:.4f}), MAE: {train_metrics['mae']:.2f} | " 
            f"Val Loss: {val_metrics['loss']:.4f} (NCOR:{val_metrics['ncor']:.4f}, MSE:{val_metrics['mse']:.4f}), MAE: {val_metrics['mae']:.2f} | "
            f"Test Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f}"
        )
    elif isinstance(criterion, Neg_Pearson) or isinstance(criterion,NCorshift):
        tqdm.write(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f} | " 
            f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f} | "
            f"Test Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f}"
        )
    else :

            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} NCOR:{train_metrics['ncor']:.4f}, kl:{train_metrics['kl']:.4f}, MAE: {train_metrics['mae']:.2f} | " 
                f"Val Loss: {val_metrics['loss']:.4f} NCOR:{val_metrics['ncor']:.4f}, kl:{val_metrics['kl']:.4f}, MAE: {val_metrics['mae']:.2f} | "
                f"Test Loss: {test_metrics['loss']:.4f} NCOR:{test_metrics['ncor']:.4f}, kl:{test_metrics['kl']:.4f}, MAE: {test_metrics['mae']:.2f}"
            )

    torch.save(model.state_dict(), os.path.join(parameter_save_dir, f'{str(lr_)}_epoch:{str(epoch+1)}_MAE:{val_metrics["mae"]: .2f}.pth'))

    if val_metrics['loss'] < best_val_loss :
        best_val_loss = val_metrics['loss']
        if 'shift_list' in val_metrics :
            np.save(os.path.join(shift_list_save_dir, f'best_val_shifts_{epoch+1}.npy') , val_metrics['shift_list'] )
        if 'shift_list' in test_metrics :
            np.save(os.path.join(shift_list_save_dir, f'best_test_shifts_{epoch+1}.npy') , test_metrics['shift_list'] )




