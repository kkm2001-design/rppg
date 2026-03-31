import os

import h5py
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import sys
from tqdm import tqdm 
from utils.utils_sig import get_sig

@staticmethod
def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-6)

    diffnormalized_data = np.clip(diffnormalized_data, -1.0, 1.0)
    # 0이 아닌 픽셀만으로 std 계산
    nonzero_mask = diffnormalized_data != 0
    if nonzero_mask.sum() > 0:
        std = np.std(diffnormalized_data[nonzero_mask])
        std = max(std, 1e-6)
    else:
        std = 1.0
    #diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = diffnormalized_data / std
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data

@staticmethod
def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

# data loading
class rPPGDataset(Dataset) :
    def __init__(self, h5_file_paths, mode='train', crop_len=150) : 
        self.data_info = []
        self.crop_len = crop_len
        self.mode = mode

        for path in h5_file_paths :
            with h5py.File(path, 'r', locking=False) as f :
                num_windows = f['video'].shape[0]
                for i in range(num_windows) :
                    self.data_info.append((path, i))
        
    def __len__(self) : ## len(train_dataset) 실행시 train_dataset.__len__() 이 실행됨
        return(len(self.data_info))

    def __getitem__(self, idx):
        file_path, window_idx = self.data_info[idx]
    
        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx].astype(np.float32)  # (T, H, W)
            gt    = f['gt_ppg'][window_idx].astype(np.float32) # (T,)
    
        T = video.shape[0]
    
        # ── temporal crop ──
        if self.crop_len is not None and self.crop_len < T:
            if self.mode == 'train':
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                start = (T - self.crop_len) // 2
            video = video[start:start + self.crop_len]  # (crop_len, H, W)
            gt    = gt[start:start + self.crop_len]
    
        # ── 시간축 stable mask: 세그먼트 내내 nonzero인 픽셀만 유지 ──
        # 하나라도 0인 프레임이 있으면 해당 픽셀 전체를 0으로 마스킹
        stable_mask = np.all(video != 0, axis=0)   # (H, W) True=항상 유효
        video = video * stable_mask[np.newaxis, :, :]  # (T, H, W)
    
        # ── diff_normalize ──
        video = np.expand_dims(video, axis=0)       # (1, T, H, W)
        video = np.transpose(video, (1, 2, 3, 0))   # (T, H, W, 1)
        video = diff_normalize_data(video)          # (T, H, W, 1)
    
        gt = diff_normalize_label(gt)
    
        video_tensor = torch.tensor(video, dtype=torch.float32)
        gt_tensor    = torch.tensor(gt,    dtype=torch.float32)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
    
        return video_tensor, gt_tensor


class rPPGDataset_1D(Dataset) :
    def __init__(self, h5_file_paths) : 
        self.data_info = []

        for path in h5_file_paths :
            with h5py.File(path, 'r', locking=False) as f :
                num_windows = f['video'].shape[0]
                for i in range(num_windows) :
                    self.data_info.append((path, i))
        
    def __len__(self) : ## len(train_dataset) 실행시 train_dataset.__len__() 이 실행됨
        return(len(self.data_info))

    def _normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __getitem__(self, idx) :  ## train_dataset[0] 실행시 train_dataset.__getitem__(0) 이 실행됨
        file_path, window_idx = self.data_info[idx]

        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx]
            gt = f['gt_ppg'][window_idx]

        sig_extractor = get_sig(video)
        signal_1d = sig_extractor.get_rl_allrois()

        signal_1d = self._normalize(signal_1d)

        signal_tensor = torch.tensor(signal_1d, dtype = torch.float32).unsqueeze(0)
        gt_tensor = torch.tensor(gt, dtype = torch.float32)

        # print(f'signal shape : {signal_tensor.shape}')
        # print(f"signal max: {signal_tensor.max()}")
        # print(f"signal min: {signal_tensor.min()}")
        # print(f"gt max: {gt_tensor.max()}")
        # print(f"gt min: {gt_tensor.min()}")
        
        return signal_tensor, gt_tensor