import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from pathlib import Path


H5_DIR   = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped'
SAVE_DIR = '/nasdata4/2yhy/rPPG_code/rPPG_datasets/raw_data/CNA_members/KM_preprocess_data/histogram_plots_raw'
os.makedirs(SAVE_DIR, exist_ok=True)

h5_files = sorted(Path(H5_DIR).glob('*.h5'))
n = len(h5_files)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.ravel()

for i, h5_path in enumerate(h5_files):
    with h5py.File(str(h5_path), 'r') as f:
        flat = f['video'][:].ravel().astype(np.float32)
        flat = (flat/255.0 -0.5) / 0.5
    axes[i].hist(flat, bins=256, color='steelblue', alpha=0.85, log=True)
    axes[i].set_title(h5_path.stem.replace('_uv', ''), fontsize=8)
    axes[i].set_xlabel('value', fontsize=7)
    axes[i].set_ylabel('count (log)', fontsize=7)
    axes[i].tick_params(labelsize=7)
    print(f"{h5_path.stem}  mean={flat.mean():.4f}  std={flat.std():.4f}  min={flat.min():.4f}  max={flat.max():.4f}")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Per-subject nir_uv_data Pixel Distribution (raw)', fontsize=13, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(SAVE_DIR, 'per_subject_histogram_raw.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n저장: {save_path}')