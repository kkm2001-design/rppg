import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import detrend, butter, filtfilt
import neurokit2 as nk
from tqdm import tqdm
import os
import glob

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
PPG_QUALITY_THR = 0.95
PPG_QUALITY_FS  = 30
GRID_ROWS       = 4
GRID_COLS       = 4


def hampel_filter(x, window_size=10, n_sigma=3.0):
    x = x.copy()
    T = len(x)
    k = 1.4826
    for i in range(T):
        lo     = max(0, i - window_size)
        hi     = min(T, i + window_size + 1)
        window = x[lo:hi]
        med    = np.median(window)
        mad    = k * np.median(np.abs(window - med))
        if mad > 0 and np.abs(x[i] - med) > n_sigma * mad:
            x[i] = med
    return x


def butter_bandpass_filter_t(data, fs=30, low=0.66, high=4.0, order=4,
                              apply_hampel=True,
                              hampel_window=10, hampel_sigma=3.0):
    nyq  = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')

    if data.ndim == 1:
        x = data.copy()
        if apply_hampel:
            x = hampel_filter(x, hampel_window, hampel_sigma)
        x = detrend(x, type='linear')
        x = filtfilt(b, a, x)
        return x.astype(np.float32), None

    T, H, W    = data.shape
    valid_mask = np.any(data != 0, axis=0)

    if valid_mask.sum() == 0:
        return data.astype(np.float32), valid_mask

    flat      = data.reshape(T, -1).astype(np.float64)
    valid_idx = valid_mask.reshape(-1)
    x         = flat[:, valid_idx]

    x = detrend(x, axis=0, type='linear')
    x = filtfilt(b, a, x, axis=0)

    out = np.zeros_like(flat, dtype=np.float32)
    out[:, valid_idx] = x.astype(np.float32)

    return out.reshape(T, H, W), valid_mask


def compute_ppg_quality(sig: np.ndarray, fs: int = PPG_QUALITY_FS) -> float:
    if len(sig) < fs * 2 or np.std(sig) < 1e-6:
        return 0.0
    try:
        quality = nk.ppg_quality(sig, sampling_rate=fs, method="templatematch")
        return float(np.mean(quality))
    except Exception:
        return 0.0


def build_stmap(video: np.ndarray,
                grid_rows: int = GRID_ROWS,
                grid_cols: int = GRID_COLS) -> np.ndarray:
    T, H, W  = video.shape
    n_blocks = grid_rows * grid_cols
    stmap    = np.zeros((n_blocks, T), dtype=np.float32)

    h_edges = np.linspace(0, H, grid_rows + 1).astype(int)
    w_edges = np.linspace(0, W, grid_cols + 1).astype(int)

    block_idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            h0, h1 = h_edges[r], h_edges[r + 1]
            w0, w1 = w_edges[c], w_edges[c + 1]
            block  = video[:, h0:h1, w0:w1]

            valid = np.any(block != 0, axis=0)
            if valid.sum() > 0:
                sig = block[:, valid].mean(axis=1)
            else:
                sig = np.zeros(T, dtype=np.float32)

            stmap[block_idx] = sig.astype(np.float32)
            block_idx += 1

    return stmap  # (16, T)


def normalize_label(label: np.ndarray) -> np.ndarray:
    std = label.std()
    if std < 1e-6:
        print("std check")
        return np.zeros_like(label, dtype=np.float32)
    return ((label - label.mean()) / std).astype(np.float32)


class rPPGDataset(Dataset):
    def __init__(self,
                 h5_file_paths,
                 mode='train',
                 crop_len=300,
                 quality_thr=PPG_QUALITY_THR):

        self.data_info   = []
        self.subj_stats  = {}
        self.crop_len    = crop_len
        self.mode        = mode
        self.quality_thr = quality_thr

        n_total   = 0
        n_skipped = 0

        for path in h5_file_paths:
            with h5py.File(path, 'r', locking=False) as f:
                num_windows = f['video'].shape[0]
                T           = f['video'].shape[1]
                gt_all      = f['gt_ppg'][:]

            # ── split 기준 ──
            # train: 세그먼트 60% 사용 (랜덤 crop)
            # val/test: 전체 세그먼트 사용 (3구간 평가)
            n_train = int(num_windows * 0.6)

            if self.mode == 'train':
                seg_indices = range(0, n_train)
            else:  # val / test: 전체
                seg_indices = range(0, num_windows)

            # ── 1. quality 캐싱: 전체 세그먼트 1회 계산 ──
            print(f"  [{os.path.basename(path)}] "
                  f"quality 계산 중 ({num_windows}개)...")
            quality_scores = {}
            for i in tqdm(range(num_windows), leave=False):
                quality_scores[i] = compute_ppg_quality(
                    gt_all[i].astype(np.float64))

            # ── 2. quality 통과한 전체 세그먼트 (치팅) ──
            all_valid_indices = [i for i in range(num_windows)
                                 if quality_scores[i] >= self.quality_thr]
            print(f"    quality 통과: "
                  f"{len(all_valid_indices)}/{num_windows}개")

            # ── 3. subject-wise stats ──
            if all_valid_indices:
                print(f"    stats 계산 중...")
                stats = self._compute_subject_stats(
                    path, all_valid_indices, gt_all)
                self.subj_stats[path] = stats
                print(f"    gt_std={stats['gt_std']:.6f}")
            else:
                self.subj_stats[path] = {'gt_std': 1.0}

            # ── 4. 현재 mode의 valid_indices ──
            valid_indices = [i for i in seg_indices
                             if quality_scores[i] >= self.quality_thr]

            n_mode_total   = len(list(seg_indices))
            n_mode_skipped = n_mode_total - len(valid_indices)
            n_total   += n_mode_total
            n_skipped += n_mode_skipped

            # ── 5. data_info 등록 ──
            for i in valid_indices:
                if self.mode == 'train':
                    # 랜덤 300프레임 crop → None으로 등록
                    self.data_info.append((path, i, None))
                else:
                    # 900프레임 → 300씩 3구간 고정 평가
                    n_splits = T // crop_len  # 900 // 300 = 3
                    for s in range(n_splits):
                        self.data_info.append((path, i, s * crop_len))

        print(f"\n[{mode}] 전체={n_total} | "
              f"제외={n_skipped} (quality<{quality_thr}) | "
              f"사용={n_total - n_skipped}\n")

    @staticmethod
    def _compute_subject_stats(h5_path, seg_indices, gt_all,
                                fs=30, low=0.66, high=4.0, order=4):
        all_gt_diffs = []
        for i in seg_indices:
            gt = gt_all[i].astype(np.float32)
            gt_bp, _ = butter_bandpass_filter_t(
                gt, fs=fs, low=low, high=high,
                order=order, apply_hampel=True)
            diff_g = np.diff(gt_bp).astype(np.float32)
            all_gt_diffs.append(diff_g)

        all_g  = np.concatenate(all_gt_diffs)
        gt_std = float(max(np.std(all_g), 1e-6))
        return {'gt_std': gt_std}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        file_path, window_idx, start = self.data_info[idx]

        with h5py.File(file_path, 'r', locking=False) as f:
            video = f['video'][window_idx].astype(np.float32)  # (900, H, W)
            gt    = f['gt_ppg'][window_idx].astype(np.float32) # (900,)

        T = video.shape[0]

        # ── Step 1: Bandpass (전체 900프레임 기준) ──
        video, _ = butter_bandpass_filter_t(video, apply_hampel=False)
        gt, _    = butter_bandpass_filter_t(gt,    apply_hampel=True)

        # ── Step 2: STMap (전체 900프레임) ──
        stmap = build_stmap(video)   # (16, 900)

        # ── Step 3: Label normalize (전체 900프레임 기준) ──
        gt = normalize_label(gt)     # (900,)

        # ── Step 4: Crop ──
        if start is None:   # train: 랜덤 300프레임
            s = np.random.randint(0, T - self.crop_len + 1)
        else:               # val/test: 고정 구간 (0, 300, 600)
            s = start

        stmap = stmap[:, s:s + self.crop_len]  # (16, 300)
        gt    = gt[s:s + self.crop_len]        # (300,)

        stmap_tensor = torch.tensor(stmap, dtype=torch.float32)  # (16, 300)
        gt_tensor    = torch.tensor(gt,    dtype=torch.float32)  # (300,)

        return stmap_tensor, gt_tensor


def build_dataloaders(train_paths, val_paths, test_paths,
                      crop_len=300,
                      quality_thr=PPG_QUALITY_THR,
                      batch_size=8,
                      num_workers=4):
    train_ds = rPPGDataset(train_paths, mode='train',
                           crop_len=crop_len, quality_thr=quality_thr)
    val_ds   = rPPGDataset(val_paths,   mode='val',
                           crop_len=crop_len, quality_thr=quality_thr)
    test_ds  = rPPGDataset(test_paths,  mode='test',
                           crop_len=crop_len, quality_thr=quality_thr)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    DATA_DIR = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_normalizeX'
    all_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))
    assert len(all_paths) > 0, "h5 경로를 확인하세요."

    # subject-wise split (12/2/2)
    train_paths = all_paths[:12]
    val_paths   = all_paths[12:14]
    test_paths  = all_paths[14:]

    print(f"train subjects: {len(train_paths)}")
    print(f"val   subjects: {len(val_paths)}")
    print(f"test  subjects: {len(test_paths)}")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_paths, val_paths, test_paths,
        crop_len=300, quality_thr=0.95, batch_size=8
    )

    stmap_b, gt_b = next(iter(train_loader))
    print(f"\nstmap batch : {stmap_b.shape}")   # (8, 16, 300)
    print(f"gt    batch : {gt_b.shape}")         # (8, 300)
    print(f"stmap range : [{stmap_b.min():.4f}, {stmap_b.max():.4f}]")
    print(f"gt    mean/std : {gt_b.mean():.4f} / {gt_b.std():.4f}")

'''

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
    #nonzero_mask = diffnormalized_data != 0
    #if nonzero_mask.sum() > 0:
    #    std = np.std(diffnormalized_data[nonzero_mask])
    #    std = max(std, 1e-6)
    #    if std==1e-6:
    #        print("std check")
    #else:
    #    std = 1.0
    #    print("std check")
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    #diffnormalized_data = diffnormalized_data / std
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

class rPPGDataset(Dataset):
    def __init__(self, h5_file_paths, mode='train', crop_len=300):
        self.data_info = []
        self.crop_len = crop_len
        self.mode = mode

        for path in h5_file_paths:
            with h5py.File(path, 'r', locking=False) as f:
                num_windows = f['video'].shape[0]
                T = f['video'].shape[1]  # 900

                for i in range(num_windows):
                    if self.mode == 'train':
                        # 기존과 동일: 세그먼트 단위 등록
                        self.data_info.append((path, i, None))
                    else:
                        # val/test: non-overlap 3구간 등록
                        n_splits = T // crop_len  # 900 // 300 = 3
                        for s in range(n_splits):
                            start = s * crop_len
                            self.data_info.append((path, i, start))
                            
    def __len__(self) : ## len(train_dataset) 실행시 train_dataset.__len__() 이 실행됨

        return(len(self.data_info))
    def __getitem__(self, idx):
        file_path, window_idx, start = self.data_info[idx]

        with h5py.File(file_path, 'r', locking=False) as f:
            if start is None:
                # train: 전체 로드 후 랜덤 crop
                video = f['video'][window_idx].astype(np.float32)
                gt    = f['gt_ppg'][window_idx].astype(np.float32)
                T = video.shape[0]
                s = np.random.randint(0, T - self.crop_len + 1)
                video = video[s:s + self.crop_len]
                gt    = gt[s:s + self.crop_len]
            else:
                # val/test: 고정 구간 슬라이싱
                video = f['video'][window_idx, start:start+self.crop_len].astype(np.float32)
                gt    = f['gt_ppg'][window_idx, start:start+self.crop_len].astype(np.float32)

        # 이하 기존과 동일
        stable_mask = np.all(video != 0, axis=0)
        video = video * stable_mask[np.newaxis, :, :]

        video = np.expand_dims(video, axis=0)
        video = np.transpose(video, (1, 2, 3, 0))
        video = diff_normalize_data(video)
        gt    = diff_normalize_label(gt)

        video_tensor = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2)
        gt_tensor    = torch.tensor(gt,    dtype=torch.float32)

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
'''