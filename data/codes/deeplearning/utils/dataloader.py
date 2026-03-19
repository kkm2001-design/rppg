import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
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
class rPPGDataset(Dataset) :
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
        return (data - 0.5) / 0.5

    def __getitem__(self, idx) :  ## train_dataset[0] 실행시 train_dataset.__getitem__(0) 이 실행됨
        file_path, window_idx = self.data_info[idx]

        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx]
            gt = f['gt_ppg'][window_idx]

        video = np.expand_dims(video, axis = 0)

        video_tensor = torch.tensor(video, dtype = torch.float32)
        gt_tensor = torch.tensor(gt, dtype = torch.float32)


        video_tensor = self._normalize(video_tensor)
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