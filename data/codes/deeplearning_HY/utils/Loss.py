'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils.post_process import calculate_metric_per_video

def bandpass_fft(x, fs=30, low=0.66, high=4.0):
    """
    x: (B, T) tensor, gradient 흐름 유지
    """
    T = x.shape[-1]
    
    # FFT
    X = torch.fft.rfft(x, dim=-1)          # (B, T//2+1) complex
    freqs = torch.fft.rfftfreq(T, d=1.0/fs).to(x.device)  # Hz
    
    # 주파수 마스크 (hard mask → gradient 차단 없음, 마스크 자체는 상수)
    mask = ((freqs >= low) & (freqs <= high)).float()  # (T//2+1,)
    
    # 마스크 적용 후 IFFT
    X_filtered = X * mask                  # broadcast: (B, T//2+1)
    x_filtered = torch.fft.irfft(X_filtered, n=T, dim=-1)  # (B, T)
    
    return x_filtered


class Neg_Pearson_Freq(nn.Module):
    def __init__(self, fs=30, low=0.66, high=4.0):
        super().__init__()
        self.fs   = fs
        self.low  = low
        self.high = high

    def forward(self, pred, gt):
        # pred, gt: (B, T)
        pred_f = bandpass_fft(pred, self.fs, self.low, self.high)
        #gt_f   = bandpass_fft(gt,   self.fs, self.low, self.high)

        # Neg Pearson
        pred_m = pred_f - pred_f.mean(dim=-1, keepdim=True)
        #gt_m   = gt_f   - gt_f.mean(dim=-1,   keepdim=True)
        gt_m = gt
        numer  = (pred_m * gt_m).sum(dim=-1)
        denom  = torch.sqrt(
            (pred_m**2).sum(dim=-1) * (gt_m**2).sum(dim=-1)
        ).clamp(min=1e-8)

        r = numer / denom          # (B,)
        return (1-r).mean()

class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = bandpass_fft(x, 30, 0.66, 4.0)
        
        x = x - torch.mean(x, dim=-1, keepdim=True)
        
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x

class PSD_MSE_Loss(nn.Module):
    """
    pred PSD와 gt PSD 간의 MSE loss
    관심 주파수 영역(high_pass ~ low_pass BPM)만 사용
    CalculateNormPSD 재활용
    """
    def __init__(self, Fs=30, high_pass=40, low_pass=250):
        super().__init__()
        self.norm_psd = CalculateNormPSD(Fs=Fs,
                                          high_pass=high_pass,
                                          low_pass=low_pass)
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, gt, zero_pad=2):
        """
        pred, gt: (B, T) or (T,)
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            gt   = gt.unsqueeze(0)

        B = pred.shape[0]
        loss = torch.tensor(0.0, device=pred.device)

        for i in range(B):
            p = pred[i]

            pred_psd = self.norm_psd(p,      zero_pad=zero_pad)
            gt_psd   = self.norm_psd(gt[i],  zero_pad=zero_pad)

            loss = loss + self.mse(pred_psd, gt_psd)

        return loss / B

class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
    
        self.criterion_Pearson = Neg_Pearson_Freq()
        self.criterion_PSD = PSD_MSE_Loss()

    def forward(self, pred_ppg, labels, epoch, FS=30, diff_flag=False):
        B = pred_ppg.shape[0]

        loss_time = torch.tensor(0.0).cuda()
        loss_hr   = torch.tensor(0.0).cuda()

        lambda_time = 1.0
        lambda_HR = 0.0

        #if epoch > 0:
        #    lambda_time = 0.1
        #    lambda_CE = 5.0
        #    lambda_HR = 5.0
        #else:
        #    lambda_time = 0.2
        #    lambda_CE = 1.0
        #    lambda_HR = 1.0

        for i in range(B):
            pred_i  = pred_ppg[i]  # (T,)
            label_i = labels[i]    # (T,)

            # Pearson
            loss_time = loss_time + self.criterion_Pearson(
                pred_i.unsqueeze(0), label_i.unsqueeze(0))

            # HR loss
            hr = self.criterion_PSD(pred_i.unsqueeze(0), label_i.unsqueeze(0))
            loss_hr = loss_hr + hr

        loss_time = loss_time / B
        loss_hr   = loss_hr   / B

        if torch.isnan(loss_time):
            loss_time = torch.tensor(0.0).cuda()

        loss = (lambda_time * loss_time) + (lambda_HR * loss_hr)
        return loss, loss_time, loss_hr

