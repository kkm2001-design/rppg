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
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
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
        return rPPG

# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================





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


# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================
# =================================================================# =================================================================# =================================================================


"""This file is a combination of Physformer.py and transformer_layer.py
   in the official PhysFormer implementation here:
   https://github.com/ZitongYu/PhysFormer

   model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import numpy as np
from typing import Optional
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import math

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        x = self.fc1(x)                      # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)                  # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)                      # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score

class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score

# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        in_channels: int = 3, 
        frame: int = 300,
        theta: float = 0.2, # CDC_T 활성화 파라미터 (1차 미분 안 하시면 유지하세요)
        image_size: Optional[int] = None,
    ):
        super().__init__()

        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              
        self.patches = patches # [수정 3-1] forward에서 동적인 뷰어 조정을 위해 패치 사이즈를 클래스 변수로 저장

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        # [수정 1] 입력 채널 변경: 3 -> 1 (IR 카메라 흑백 영상 지원)
        self.Stem0 = nn.Sequential(
            nn.Conv3d(1, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        # [수정 2] 64x64 입력을 4x4 패치로 맞추기 위해 MaxPool3d 제거
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)), <-- 삭제 (차원 유지용)
        )
            
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp):

        # b is batch number, c channels, t frame, fh frame height, and fw frame width
        b, c, t, fh, fw = x.shape
        
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64] -> Maxpool을 지웠으므로 공간 해상도가 유지됨
        
        x = self.patch_embedding(x)  # [B, 64, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]
        
        Trans_features, Score1 =  self.transformer1(x, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features2, Score2 =  self.transformer2(Trans_features, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features3, Score3 =  self.transformer3(Trans_features2, gra_sharp)  # [B, 4*4*40, 64]
        
        # [수정 3-2] 900프레임 등 가변적인 길이에 대응하기 위해 t//4 하드코딩을 t//ft 로 변경
        ft, fh, fw = as_tuple(self.patches)
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//ft, 4, 4) 
        
        features_last = self.upsample(features_last)            # x [B, 64, 7*7, 80]
        features_last = self.upsample2(features_last)            # x [B, 32, 7*7, 160]
        
        features_last = torch.mean(features_last,3)     # x [B, 32, 160, 4]  
        features_last = torch.mean(features_last,3)     # x [B, 32, 160]    
        rPPG = self.ConvBlockLast(features_last)    # x [B, 1, 160]
        
        rPPG = rPPG.squeeze(1)
        
        return rPPG, Score1, Score2, Score3