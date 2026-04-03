import os
import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F


# =================================================================# =================================================================# =================================================================
# physnet 모델
class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=64):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        # self.fusion_stem = fusion_stem()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(1, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((None , 1, 1))

    def forward(self, x):

        # The standardization of the raw frames yields poor results; using frame differences as input instead.
        #x = x.permute(0,2,1,3,4)
        #x = x[:,:3,:,:,:]
        
        # w. fusion_stem (set the input as standardized data in config file)
        # N, D, C, H, W = x.shape
        # x = self.fusion_stem(x)
        # x = x.view(N,D,32,H//2,W//2).permute(0,2,1,3,4)  # x [32, T, 64,64]

        x_visual = x
        [batch, channel, length, width, height] = x.shape
        #print("input : ",x.shape)

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]
        #print("maxpool1 : ",x.shape)
        x = self.ConvBlock2(x)  # x [32, T, 64,64]


        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)
        #print("maxpool2 : ",x.shape)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]
        #print("maxpool3 : ",x.shape)
        
        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]
        #print("maxpool4 : ",x.shape)

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        #print("end of Encoder : ",x.shape)
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        #print("Decoder1 : ",x.shape)
        x = self.upsample2(x)  # x [64, T, 8, 8]
        #print("Decoder2 : ",x.shape)

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        #print("final : ",x.shape)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)
        #print("final22 : ",x.shape)
        #print(sjdak)
        return rPPG, x_visual, x_visual3232, x_visual1616

# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
'''

# =================================================================# =================================================================# =================================================================
# physnet 모델
class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=64):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        # self.fusion_stem = fusion_stem()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(1, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.InstanceNorm3d(16,affine=True),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(32,affine=True),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.InstanceNorm3d(64,affine=True),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.InstanceNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.InstanceNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((None , 1, 1))

    def forward(self, x):

        # The standardization of the raw frames yields poor results; using frame differences as input instead.
        #x = x.permute(0,2,1,3,4)
        #x = x[:,:3,:,:,:]
        
        # w. fusion_stem (set the input as standardized data in config file)
        # N, D, C, H, W = x.shape
        # x = self.fusion_stem(x)
        # x = x.view(N,D,32,H//2,W//2).permute(0,2,1,3,4)  # x [32, T, 64,64]

        x_visual = x
        [batch, channel, length, width, height] = x.shape
        #print("input : ",x.shape)

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]
        #print("maxpool1 : ",x.shape)
        x = self.ConvBlock2(x)  # x [32, T, 64,64]


        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)
        #print("maxpool2 : ",x.shape)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]
        #print("maxpool3 : ",x.shape)
        
        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]
        #print("maxpool4 : ",x.shape)

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        #print("end of Encoder : ",x.shape)
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        #print("Decoder1 : ",x.shape)
        x = self.upsample2(x)  # x [64, T, 8, 8]
        #print("Decoder2 : ",x.shape)

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        #print("final : ",x.shape)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)
        #print("final22 : ",x.shape)
        #print(sjdak)
        return rPPG, x_visual, x_visual3232, x_visual1616

# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
'''


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



# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================




#1d conv model

class ConvBlock1D(nn.Module):
    """Conv → BN → ReLU 두 번"""
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        # ── Encoder ──────────────────────────────
        self.enc1 = ConvBlock1D(in_ch,      base_ch)      # (B,32,900)
        self.enc2 = ConvBlock1D(base_ch,    base_ch*2)    # (B,64,450)
        self.enc3 = ConvBlock1D(base_ch*2,  base_ch*4)    # (B,128,225)
        self.pool = nn.MaxPool1d(2)

        # ── Bottleneck ────────────────────────────
        self.bottleneck = ConvBlock1D(base_ch*4, base_ch*8)  # (B,256,112)

        # ── Decoder ──────────────────────────────
        # Upsample 후 skip과 concat → 채널 두 배 들어옴
        self.up3   = nn.ConvTranspose1d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3  = ConvBlock1D(base_ch*8, base_ch*4)   # cat → 256ch 입력

        self.up2   = nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2  = ConvBlock1D(base_ch*4, base_ch*2)   # cat → 128ch 입력

        self.up1   = nn.ConvTranspose1d(base_ch*2, base_ch,   kernel_size=2, stride=2)
        self.dec1  = ConvBlock1D(base_ch*2, base_ch)     # cat → 64ch 입력

        # ── Output head ──────────────────────────
        self.head  = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, 900)

        # Encoder
        s1 = self.enc1(x)               # (B, 32, 900)
        s2 = self.enc2(self.pool(s1))   # (B, 64, 450)
        s3 = self.enc3(self.pool(s2))   # (B, 128, 225)

        # Bottleneck
        b  = self.bottleneck(self.pool(s3))  # (B, 256, 112)

        # Decoder + skip connection
        d3 = self.up3(b)                          # (B, 128, 224)
        d3 = self._match_and_cat(d3, s3)          # (B, 256, 225)
        d3 = self.dec3(d3)                        # (B, 128, 225)

        d2 = self.up2(d3)                         # (B, 64, 450)
        d2 = self._match_and_cat(d2, s2)          # (B, 128, 450)
        d2 = self.dec2(d2)                        # (B, 64, 450)

        d1 = self.up1(d2)                         # (B, 32, 900)
        d1 = self._match_and_cat(d1, s1)          # (B, 64, 900)
        d1 = self.dec1(d1)                        # (B, 32, 900)

        out = self.head(d1)                       # (B, 1, 900)
        return out.squeeze(1)                     # (B, 900)

    def _match_and_cat(self, x, skip):
        """Upsample 후 길이가 1 차이날 수 있어서 맞춰주는 처리"""
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1])
        return torch.cat([x, skip], dim=1)