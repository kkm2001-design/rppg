import os
import torch
import torch.nn as nn 
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

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
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2)))+ 1e-8)
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss




class HybridLoss(nn.Module):
    def __init__(self, ncor_weight=1.0, mse_weight=1.0):
        super(HybridLoss, self).__init__()
        self.ncor_weight = ncor_weight
        self.mse_weight = mse_weight
        
        self.ncor_criterion = Neg_Pearson() 
        self.mse_criterion = nn.MSELoss()

    def forward(self, outputs, targets):
        loss_ncor = self.ncor_criterion(outputs, targets)
        loss_mse = self.mse_criterion(outputs, targets)
        
        # 두 Loss를 가중치와 함께 합산
        total_loss = (self.ncor_weight * loss_ncor) + (self.mse_weight * loss_mse)
        return total_loss, loss_ncor, loss_mse



'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/blob/main/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''

def kl_loss(inputs, labels):
    """
    [Shape 흐름]
    inputs: (B, K) - 모델이 예측한 확률 분포
    labels: (B, K) - 정답 가우시안 분포
    
    [수정 사항] 
    1. Double Softmax 버그 완벽 제거 (순수 torch.log 사용)
    2. Batch 단위 학습을 위해 reduction='batchmean' 사용 (배치 크기에 무관하게 일정한 Loss 스케일 유지)
    """
    criterion = nn.KLDivLoss(reduction='batchmean')
    # log(0)으로 인한 NaN 방지를 위해 1e-7 추가
    loss = criterion(torch.log(inputs + 1e-7), labels)
    return loss

class physformerLoss(object):
    
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        """
        이산 푸리에 변환(DFT) 병렬 처리 엔진
        
        [Shape 흐름]
        output: (B, N) -> (B, 1, N)
        k: (1, K) -> (1, K, 1)
        two_pi_n_over_N: (N) -> (1, 1, N)
        """
        device = output.device
        B = output.shape[0]  # 배치 사이즈 추출 (예: 4)

        # 시간 축(위상) 좌표 생성
        two_pi_n_over_N = (2 * math.pi * torch.arange(0, N, dtype=torch.float, device=device)) / N
        
        # 해닝 윈도우 생성 및 브로드캐스팅 곱셈: (B, N) * (1, N) = (B, N)
        hanning = torch.from_numpy(np.hanning(N)).type(torch.float).to(device).view(1, N)
        output = output * hanning
        
        # 병렬 내적을 위한 3차원 확장
        output = output.view(B, 1, N)           # (B, 1, N)
        k = k.type(torch.float).to(device).view(1, -1, 1)  # (1, K, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, N)    # (1, 1, N)
        
        # 브로드캐스팅 내적 연산: (B, 1, N) * (1, K, N) -> (B, K, N) -> 합산 후 (B, K)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                         + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute  # (B, K)

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        """
        주파수 인덱스(k) 변환기
        
        [수정 사항]
        기존의 1차원 찌그러뜨리기(view(1, -1)) 및 강제 Softmax 정규화 삭제.
        입력된 (B, N) 형태를 그대로 엔진에 전달하고, 반환된 Logit을 그대로 넘김.
        """
        N = output.shape[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # 반환 Shape: (B, K)
        return physformerLoss.compute_complex_absolute_given_k(output, k, N)

    @staticmethod
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std=1.0):
        """
        Soft Label (KL) + Hard Label (CE) 통합 Loss 계산기
        
        [Shape 흐름]
        inputs: (B, N) - 시간 축 rPPG 신호
        target: (B,)   - GT 심박수 (예: [65.0, 72.0, 68.0, 80.0])
        """
        B = inputs.shape[0]
        device = inputs.device
        
        # 탐색할 주파수 대역 (예: 40~150 BPM, 총 110칸)
        bpm_range = torch.arange(40, 150, dtype=torch.float, device=device)
        K = bpm_range.shape[0]

        # 1. GPU 브로드캐스팅으로 가우시안 정답 분포 생성: (B, 1) & (1, K) -> (B, K)
        target_tensor = target.view(B, 1).type(torch.float)
        bpm_range_tensor = bpm_range.view(1, K)
        
        target_distribution = torch.exp(-((bpm_range_tensor - target_tensor)**2) / (2 * std**2))
        target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True)
        target_distribution = torch.clamp(target_distribution, min=1e-15)

        # 2. 푸리에 변환 엔진 가동 (Logit 반환): (B, K)
        ca = physformerLoss.complex_absolute(inputs, Fs, bpm_range)
        
        # 3. KL Loss 계산 (ca를 임시로 확률로 변환)
        fre_distribution = ca / torch.sum(ca, dim=-1, keepdim=True)
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)
        
        # 4. CE Loss 계산 (PyTorch CE는 내부적으로 Logit을 알아서 처리함)
        # target_idx Shape: (B,)
        target_idx = (target.view(-1) - bpm_range[0]).type(torch.long)
        loss_ce = F.cross_entropy(ca, target_idx)
        
        # 5. MAE 오차 계산 (각 배치의 피크 위치 탐색)
        # ca.max(dim=-1) 결과: (B,)
        _, whole_max_idx = ca.max(dim=-1)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        # 각 피험자별 절대 오차의 평균 반환: (B,) -> 스칼라 평균
        mae_error = torch.abs(target.view(-1) - bpm_range[0] - whole_max_idx).mean()

        return loss_distribution_kl, loss_ce, mae_error

    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        """
        단일 Cross Entropy 전용 Loss 계산기 (Batch 병렬 최적화 버전)
        """
        device = inputs.device
        bpm_range = torch.arange(40, 150, dtype=torch.float, device=device)

        # ca Shape: (B, K)
        ca = physformerLoss.complex_absolute(inputs, Fs, bpm_range)

        # whole_max_idx Shape: (B,)
        _, whole_max_idx = ca.max(dim=-1)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        # target_idx Shape: (B,)
        target_idx = (target.view(-1) - bpm_range[0]).type(torch.long)
        
        loss_ce = F.cross_entropy(ca, target_idx)
        mae_error = torch.abs(target.view(-1) - bpm_range[0] - whole_max_idx).mean()
        
        return loss_ce, mae_error
        