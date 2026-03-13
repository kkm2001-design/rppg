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
from models import *
import argparse, os
import random
import math
from torchvision import transforms
from torch import nn

# data loading
class rPPGDataset(Dataset) :
    def __init__(self, h5_file_paths) : 
        self.data_info = []

        for path in h5_file_paths :
            with h5py.File(path, 'r', locking=False) as f :
                num_windows = f['video'].shape[0]
                for i in range(num_windows) :
                    self.data_info.append((path, i))
        
    def __len__(self) : ## len(train_dataset) 실행시 train_dataset.__len__() 이 실행됨
        return(len(self.data_info))

    def __getitem__(self, idx) :  ## train_dataset[0] 실행시 train_dataset.__getitem__(0) 이 실행됨
        file_path, window_idx = self.data_info[idx]

        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx]
            gt = f['gt_ppg'][window_idx]

        video = np.expand_dims(video, axis = 0)

        video_tensor = torch.tensor(video, dtype = torch.float32)
        gt_tensor = torch.tensor(gt, dtype = torch.float32)
        
        return video_tensor, gt_tensor


data_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/data_without_outlier'
all_files = sorted(glob.glob(os.path.join(data_dir, 'final_dataset_*.h5')))
all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('.h5') ]

train_files = all_files[4:]
test_files = all_files[2:4]
valid_files = all_files[:2]

train_dataset = rPPGDataset(train_files)
valid_dataset = rPPGDataset(valid_files)
test_dataset = rPPGDataset(test_files)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True, num_workers =4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for inputs, targets in train_loader:
    print("\n[DataLoader Test]")
    print(f"Input Video Batch Shape: {inputs.shape}")  # 예상: [4, 1, 900, 64, 64]
    print(f"Target GT Batch Shape: {targets.shape}")   # 예상: [4, 900]
    break

            

class Neg_Pearson(nn.Module):
    """
    The Neg_Pearson Module is from the orignal author of Physnet.
    Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
    source: https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
    """
    
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return


    def forward(self, preds, labels):       
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])             
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss



# 1. validate 함수 수정 [cite: 32, 33]
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader: 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_,_,_  = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
    return running_loss / len(loader)

# 2. train_one_epoch 함수 수정 [cite: 29, 30]
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs,_,_,_ = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

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

for epoch in main_pbar:
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, valid_loader, criterion, device)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)

    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     #torch.save(model.state_dict(), f'3dres_data_wo_ol_lr:{str(lr_)}_epoch:{str(epoch)}.pth')




