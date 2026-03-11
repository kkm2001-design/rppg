import os
import torch
import torch.nn as nn 
import numpy as np


# =================================================================
# 1.3D_Resnet 
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
        x = self.regressor(x) 
        
        # 4. 불필요한 차원(껍데기) 제거: [B, 1, 900, 1, 1] -> [B, 900]
        x = x.squeeze(-1).squeeze(-1).squeeze(1) 
        
        return x
