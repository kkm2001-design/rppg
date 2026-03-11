import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import sys
from tqdm import tqdm 


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


data_dir = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/data_for_dl'
all_files = sorted(glob.glob(os.path.join(data_dir, 'final_dataset_*.h5')))
all_files = [os.path.join(data_dir, u) for u in sorted(os.listdir(data_dir)) if u.endswith('.h5') ]

train_files = all_files[4:]
valid_files = all_files[2:4]
test_files = all_files[:2]

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

            
        
# =================================================================
# 1. Resnet skip connection 기본 블록
# =================================================================

class ResidualBlock3D(nn.Module) :
    def __init__(self, in_channels, out_channels, spatial_stride = 1) :
        super().__init__()

        stride = (1, spatial_stride, spatial_stride) # 1자리에 들어가는 파라미터가 T축 압축을 결정 1로 설정했으므로 T 차원에서의 변화는 없음

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size =(3,3,3), stride = stride, padding = 1, bias =False) # kernel_size : (T,H,W) (T: 시간축에서 평균내는 프레임의 수)  /  input shape : (900,in_channels,H,W) > output shape : (900, out_channels, )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = (3,3,3), stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm3d(out_channels) 

        self.downsample = None 
        if in_channels != out_channels or spatial_stride != 1 : ## input값이 residual block 값과 shape달라질 시 수행
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False) ,
                                            nn.BatchNorm3d(out_channels))  

    def forward(self, x) :
        identity = x
        if self.downsample is not None :
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 스킵 커넥션 (ResNet의 핵심)
        out = self.relu(out)
        
        return out


# =====================================================================
# 2. rPPG 전용 3D-ResNet 메인 네트워크
# =====================================================================
class rPPG_3D_ResNet(nn.Module):
    def __init__(self):
        super(rPPG_3D_ResNet, self).__init__()
        
        # [Step 1] 초기 특징 추출
        # 입력: [B, 1, 900, 64, 64] (NIR 1채널) -> 출력: [B, 16, 900, 64, 64]
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # [Step 2] ResNet 블록 통과 (공간 해상도만 점진적으로 절반씩 축소)
        # 시간축 900은 그대로 유지됩니다.
        self.layer1 = ResidualBlock3D(16, 32, spatial_stride=2) # -> [B, 32, 900, 32, 32]
        self.layer2 = ResidualBlock3D(32, 64, spatial_stride=2) # -> [B, 64, 900, 16, 16]
        self.layer3 = ResidualBlock3D(64, 64, spatial_stride=2) # -> [B, 64, 900, 8, 8]
        
        # [Step 3] 공간 평균 풀링 (Spatial Global Average Pooling)
        # 8x8 얼굴 공간의 특징들을 하나로 평균 내어 순수한 시간의 흐름만 남깁니다.
        # -> [B, 64, 900, 1, 1]
        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        self.dropout = nn.Dropout(p=0.5)
        # [Step 4] 1D 맥파(rPPG) 파형 변환용 Regressor (1x1x1 Conv)
        # 64가닥의 단서들을 조합하여 최종 1개의 파형으로 만듭니다.
        # -> [B, 1, 900, 1, 1]
        self.regressor = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        # 1. 초기 컨볼루션
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 2. ResNet 블록 통과
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 3. 공간 뭉개기 및 1채널 압축
        x = self.adaptive_pool(x) 
        x = self.dropout(x)
        x = self.regressor(x) 
        
        # 4. 불필요한 차원(껍데기) 제거: [B, 1, 900, 1, 1] -> [B, 900]
        x = x.squeeze(-1).squeeze(-1).squeeze(1) 
        
        return x

class NegCorLoss(nn.Module) :
    def __init__(self) :
        super().__init__()
        
    def forward(self, preds, targets) :
        preds_norm = preds - torch.mean(preds, dim = -1, keepdim = True)
        targets_norm = targets - torch.mean(targets, dim = -1, keepdim = True)

        correlation =  torch.sum(preds_norm * targets_norm, dim = -1) / (torch.sqrt(torch.sum(preds_norm ** 2, dim=-1)) * torch.sqrt(torch.sum(targets_norm ** 2, dim=-1)) + 1e-8 )
        return  torch.mean(1 - correlation)

# 1. validate 함수 수정 [cite: 32, 33]
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader: 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
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
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(loader)

# 설정 부분은 그대로 유지
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = rPPG_3D_ResNet().to(device) 
criterion = NegCorLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_epochs = 50

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
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_rppg_model.pth')
