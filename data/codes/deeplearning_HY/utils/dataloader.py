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

    nonzero_mask = data != 0
    if nonzero_mask.sum() > 0:
        std = np.std(data[nonzero_mask])
        std = max(std, 1e-7)
    else:
        std = 1.0
        
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)

    diffnormalized_data = np.clip(diffnormalized_data, -1.0, 1.0)
    # 0이 아닌 픽셀만으로 std 계산
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
            video = f['video'][window_idx]      # expected: (T, H, W)
            gt = f['gt_ppg'][window_idx]        # expected: (T,)

        # uint8 -> float32, [0,255] -> [0,1]
        video = video.astype(np.float32)
        gt = gt.astype(np.float32)

        T = video.shape[0]

        # -----------------------------
        # temporal crop
        # -----------------------------
        if self.crop_len is not None and self.crop_len < T:
            if self.mode == 'train':
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                # val/test: deterministic center crop
                start = (T - self.crop_len) // 2

            end = start + self.crop_len
            video = video[start:end]   # (crop_len, H, W)
            gt = gt[start:end]         # (crop_len,)

        # (T, H, W) -> (1, T, H, W)
        video = np.expand_dims(video, axis=0)
        video = np.transpose(video, (1,2,3,0))

        #print(f"video max: {video.max()}")
        #print(f"video min: {video.min()}")
        #print(f"gt max: {gt.max()}")
        #print(f"gt min: {gt.min()}")

        video = diff_normalize_data(video)
        gt = diff_normalize_label(gt)

        video_tensor = torch.tensor(video, dtype=torch.float32)
        gt_tensor = torch.tensor(gt, dtype=torch.float32)
        video_tensor = video_tensor.permute(3,0,1,2)
        # -----------------------------
        # normalized frame difference
        # input shape: (C, T, H, W)
        # -----------------------------

        #print(f"video max: {video_tensor.max()}")
        #print(f"video min: {video_tensor.min()}")
        #print(f"gt max: {gt_tensor.max()}")
        #print(f"gt min: {gt_tensor.min()}")
        
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