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

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    #criterion = nn.KLDivLoss(reduce=False)
    criterion = nn.KLDivLoss(reduction='none')  # ← 수정
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson_Freq()

    def forward(self, pred_ppg, labels, epoch, FS=30, diff_flag=True):
        B = pred_ppg.shape[0]
        
        loss_time = torch.tensor(0.0).cuda()
        loss_CE   = torch.tensor(0.0).cuda()
        loss_hr   = torch.tensor(0.0).cuda()
    
        for i in range(B):
            pred_i  = pred_ppg[i]  # (T,)
            label_i = labels[i]    # (T,)
    
            # Pearson
            loss_time = loss_time + self.criterion_Pearson(
                pred_i.unsqueeze(0), label_i.unsqueeze(0))
    
            # Frequency loss: 1개씩
            ce, _ = TorchLossComputer.Frequency_loss(
                pred_i, label_i, diff_flag=diff_flag, Fs=FS, std=3.0)
            loss_CE = loss_CE + ce
    
            # HR loss: 1개씩
            hr = TorchLossComputer.HR_loss(
                pred_i, label_i, diff_flag=diff_flag, Fs=FS, std=3.0)
            loss_hr = loss_hr + hr
    
        loss_time = loss_time / B
        loss_CE   = loss_CE   / B
        loss_hr   = loss_hr   / B
    
        if torch.isnan(loss_time):
            loss_time = torch.tensor(0.0).cuda()
    
        loss = (0.2 * loss_time) + (1.0 * loss_CE) + (1.0 * loss_hr)
        return loss, loss_time, loss_CE, loss_hr

class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator


    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)


    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx

    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_gt, pred_hr_peak, SNR, macc = calculate_metric_per_video(inputs.detach().cpu(), target.detach().cpu(), diff_flag = diff_flag, fs=Fs, hr_method='FFT')
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(45, 150, dtype=torch.float).to(torch.device('cuda'))
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        hr_gt = torch.tensor(hr_gt-45).view(1).type(torch.long).to(torch.device('cuda'))
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)

    
    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        psd_gt, psd_pred, SNR, macc = calculate_metric_per_video(inputs.detach().cpu(), target.detach().cpu(), diff_flag = diff_flag, fs=Fs, hr_method='FFT')
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        return kl_loss(pred_distribution , target_distribution)

    '''

    @staticmethod
    def HR_loss(inputs, target, diff_flag, Fs, std):
        hr_label, hr_pred, SNR, macc = calculate_metric_per_video(
            inputs.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            diff_flag=diff_flag, fs=Fs, hr_method='FFT'
        )
        # hr_label, hr_pred: BPM 단위 float (예: 72.5)
        # bpm_range 45~150 기준 index로 변환
        hr_pred_idx  = float(hr_pred)  - 45
        hr_label_idx = float(hr_label) - 45

        # clamp: 범위 밖이면 경계로
        hr_pred_idx  = max(0, min(104, hr_pred_idx))
        hr_label_idx = max(0, min(104, hr_label_idx))

        pred_distribution = [
            normal_sampling(hr_pred_idx, i, std) 
            for i in range(105)   # 45~150 → 105개
        ]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).cuda()

        target_distribution = [
            normal_sampling(hr_label_idx, i, std)
            for i in range(105)
        ]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).cuda()

        return kl_loss(pred_distribution, target_distribution)
    '''

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
        loss = 0.0
        preds_norm = (preds - preds.mean(dim=-1, keepdim=True)) / \
                (preds.std(dim=-1, keepdim=True) + 1e-6)

        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])             
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            numer = (N*sum_xy - sum_x*sum_y)
            denom = (torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            pearson = numer/denom
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss

def bandpass_fft(x, fs=30, low=0.7, high=4.0):
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
        gt_f   = bandpass_fft(gt,   self.fs, self.low, self.high)

        # Neg Pearson
        pred_m = pred_f - pred_f.mean(dim=-1, keepdim=True)
        gt_m   = gt_f   - gt_f.mean(dim=-1,   keepdim=True)

        numer  = (pred_m * gt_m).sum(dim=-1)
        denom  = torch.sqrt(
            (pred_m**2).sum(dim=-1) * (gt_m**2).sum(dim=-1)
        ).clamp(min=1e-8)

        r = numer / denom          # (B,)
        return (-r).mean()
'''

class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):
            # pred z-score 정규화
            p = preds[i]
            p = (p - p.mean()) / (p.std() + 1e-8)

            l = labels[i]
            l = (l - l.mean()) / (l.std() + 1e-8)

            N      = p.shape[0]
            sum_x  = torch.sum(p)
            sum_y  = torch.sum(l)
            sum_xy = torch.sum(p * l)
            sum_x2 = torch.sum(p ** 2)
            sum_y2 = torch.sum(l ** 2)

            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt(
                    (N * sum_x2 - sum_x ** 2) *
                    (N * sum_y2 - sum_y ** 2)
                ) + 1e-8
            )
            loss += 1 - pearson

        return loss / preds.shape[0]


class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss

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
'''