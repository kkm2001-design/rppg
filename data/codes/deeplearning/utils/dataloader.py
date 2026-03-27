import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
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

class rPPGDataset(Dataset):
    def __init__(self, h5_file_paths, norm_type='fixed'):
        """
        video는 UV registration 결과로 이미 [0, 1] 스케일링된 상태
        gt_ppg는 이미 bandpass filter + z-norm 완료된 상태 → 건드리지 않음
 
        norm_type:
            'fixed'  : (x - 0.5) / 0.5  → [-1, 1]  (baseline, [0,1] 입력에 적합)
            'zscore' : (x - mean) / std  → window-level 분포 정규화 (비교 실험용)
        """
        self.data_info = []
        self.norm_type  = norm_type
 
        for path in h5_file_paths:
            with h5py.File(path, 'r', locking=False) as f:
                num_windows = f['video'].shape[0]
                for i in range(num_windows):
                    self.data_info.append((path, i))
 
    def __len__(self):
        return len(self.data_info)
 
    def _normalize_video(self, video: np.ndarray) -> np.ndarray:
        """
        video: (T, H, W), 이미 [0, 1] 범위
        """
        if self.norm_type == 'fixed':
            # [0, 1] → [-1, 1]
            return (video - 0.5) / 0.5
 
        elif self.norm_type == 'zscore':
            # window 전체 픽셀 기준 z-norm
            mean = video.mean()
            std  = video.std() + 1e-8
            return (video - mean) / std
 
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
 
    def __getitem__(self, idx):
        file_path, window_idx = self.data_info[idx]
 
        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx]   # (T, H, W), float32, [0, 1]
            gt    = f['gt_ppg'][window_idx]  # (T,), float64, already z-normed
 
        video = self._normalize_video(video)   # (T, H, W)
        video = np.expand_dims(video, axis=0)  # (1, T, H, W)
 
        video_tensor = torch.tensor(video, dtype=torch.float32)
        gt_tensor    = torch.tensor(gt,    dtype=torch.float32)
        # gt_ppg는 이미 전처리 완료 → 추가 처리 없음

        # print(f'video shape : {video_tensor.shape}')
        # print(f"video max: {video_tensor.max()}")
        # print(f"video min: {video_tensor.min()}")
        # print(f"gt max: {gt_tensor.max()}")
        # print(f"gt min: {gt_tensor.min()}")
 
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