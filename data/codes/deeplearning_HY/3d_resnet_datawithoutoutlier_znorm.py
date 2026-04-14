import os
# torch import 전에 반드시 실행
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import sys
from tqdm import tqdm 
#from utils.models import *
from utils.model_transformer import *
import argparse, os
import random
import math
from torchvision import transforms
from torch import nn
from utils.utils_sig import *
from utils.Loss import *
from utils.train import *
from utils.dataloader import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id  # worker마다 고유 시드
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# ── 설정 (이름에 반영할 하이퍼파라미터 한곳에 모음) ──
SEED = 22
set_seed(SEED)


MODEL_NAME = 'rPPGTransformer'   # 'PhysNet' 등으로 변경
LR         = 1e-2
WD         = 1e-2                # AdamW weight decay
CROP_LEN   = 300
BATCH_SIZE = 4
NUM_EPOCHS = 30

# ── DataLoader에 worker_init_fn 추가 ──
data_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_normalizeX_filtered_handcraft'

all_files = sorted(glob.glob(os.path.join(data_dir, 'final_dataset_*.h5')))
all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('.h5') ]

train_files = all_files[0:12]
valid_files = all_files[13:14]
test_files = all_files[14:16]

train_dataset = rPPGDataset(train_files, mode='train', crop_len=300)
valid_dataset = rPPGDataset(valid_files, mode='val', crop_len=300)
test_dataset = rPPGDataset(test_files, mode='test', crop_len=300)
print(f"train: {len(train_dataset)}샘플")
print(f"val:   {len(valid_dataset)}샘플")   # N_seg * 0.2 * 3구간
print(f"test:  {len(test_dataset)}샘플")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, worker_init_fn=worker_init_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, worker_init_fn=worker_init_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, worker_init_fn=worker_init_fn)

# ── 모델 / 옵티마이저 ──
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = rPPGTransformer().to(device)
criterion = RhythmFormer_Loss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

# ── 저장 prefix ──
RUN_TAG = f'{MODEL_NAME}_lr{str(LR).replace(".", "p")}_wd{str(WD).replace(".", "p")}_seed{SEED}_crop{CROP_LEN}'
# 예: rPPGTransformer_lr0.01_wd0.0001_seed42_crop300

# ── 학습 루프 ──
best_val_loss = float('inf')
train_loss_list = []
valid_loss_list = []
train_mae_list, valid_mae_list = [], []
test_mae_list = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_mae = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
    val_loss, val_mae, val_loss_time, val_loss_HR = validate(model, valid_loader, criterion, device)
    test_loss, test_mae, _, _ = validate(model, test_loader, criterion, device)

    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)
    train_mae_list.append(train_mae)
    valid_mae_list.append(val_mae)
    test_mae_list.append(test_mae)

    tqdm.write(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} BPM | "
        f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f} BPM | "
        f"Val Time: {val_loss_time:.4f}, HR: {val_loss_HR:.4f} | "
        f"Test MAE: {test_mae:.2f} BPM"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = f'{RUN_TAG}_epoch{epoch}_valloss{val_loss:.4f}'.replace('.', 'p') + '.pth'
        torch.save(model.state_dict(), save_path)
        tqdm.write(f"  ✓ Saved: {save_path}")