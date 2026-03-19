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

    def _normalize(self, data):
        return (data - 0.5) / 0.5

    def _znormalize(self, data):
        return (data - data.mean()) / (data.std()+1e-9)

    def __getitem__(self, idx):
        file_path, window_idx = self.data_info[idx]

        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx]      # expected: (T, H, W)
            gt = f['gt_ppg'][window_idx]        # expected: (T,)

        # uint8 -> float32, [0,255] -> [0,1]
        video = video.astype(np.float32) / 255.0
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

        video_tensor = torch.tensor(video, dtype=torch.float32)
        gt_tensor = torch.tensor(gt, dtype=torch.float32)

        # -----------------------------
        # normalized frame difference
        # input shape: (C, T, H, W)
        # -----------------------------
        diff = (video_tensor[:, 1:] - video_tensor[:, :-1]) / \
               (video_tensor[:, 1:] + video_tensor[:, :-1] + 1e-9)   # (C, T-1, H, W)

        # first valid diff copy -> keep same T
        first_diff = diff[:, :1].clone()
        video_tensor = torch.cat([first_diff, diff], dim=1)           # (C, T, H, W)

        # z-score normalize after diff
        video_tensor_ = self._znormalize(video_tensor)

        #print(f"video max: {video_tensor.max()}")
        #print(f"video min: {video_tensor.min()}")
        #print(f"gt max: {gt_tensor.max()}")
        #print(f"gt min: {gt_tensor.min()}")
        
        return video_tensor_, gt_tensor


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