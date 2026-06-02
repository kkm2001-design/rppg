import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr
import neurokit2 as nk

import sys
sys.path.append('/home/son/orientation-uv-rppg/data/codes/deeplearning/utils')
from models import (PhysNet_padding_Encoder_Decoder_MAX,
                    ViT_ST_ST_Compact3_TDC_gra_sharp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 필터 + HR/RMSSD (10초 윈도우, robust)
# ============================================================
def bandpass_fft_soft(signal, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    is_numpy = isinstance(signal, np.ndarray)
    if is_numpy:
        signal = torch.tensor(signal, dtype=torch.float32)
    N = signal.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(signal.device)
    fft_signal = torch.fft.rfft(signal)
    mask = torch.sigmoid(steepness * (freqs - low_hz)) * \
           torch.sigmoid(steepness * (high_hz - freqs))
    filtered = torch.fft.irfft(fft_signal * mask, n=N)
    return filtered.cpu().numpy() if is_numpy else filtered


def compute_hr_rmssd(sig, fps=30, use_bandpass=True, min_beats=6):
    """10초 윈도우용. HR + RMSSD만. robust filtering. Returns dict or None."""
    try:
        if use_bandpass:
            sig = bandpass_fft_soft(sig, fs=fps)
        clean = nk.ppg_clean(sig.copy(), sampling_rate=fps, method='elgendi')
        peaks = nk.ppg_findpeaks(clean, sampling_rate=fps)["PPG_Peaks"]
        if len(peaks) < min_beats:
            return None
        ibi_ms = np.diff(peaks) / fps * 1000
        ibi_ms = ibi_ms[(ibi_ms > 300) & (ibi_ms < 1500)]       # 40~200 BPM
        if len(ibi_ms) < 2:
            return None
        med = np.median(ibi_ms)
        ibi_ms = ibi_ms[(ibi_ms > 0.5*med) & (ibi_ms < 1.5*med)]  # artifact 제거
        if len(ibi_ms) < min_beats - 1:
            return None
        diff = np.diff(ibi_ms)
        return {'mean_hr': 60000.0/np.mean(ibi_ms),
                'rmssd':   np.sqrt(np.mean(diff**2)),
                'num_beats': len(ibi_ms) + 1}
    except Exception:
        return None


# ============================================================
# 1. 모델 빌더
# ============================================================
class PhysFormerWrapper(nn.Module):
    def __init__(self, base_model, gra_sharp=2.0):
        super().__init__()
        self.model = base_model
        self.gra_sharp = gra_sharp
    def forward(self, x):
        rPPG, _, _, _ = self.model(x, self.gra_sharp)
        return rPPG


def build_model(model_type):
    if model_type == 'physnet':
        return PhysNet_padding_Encoder_Decoder_MAX().to(device)
    elif model_type == 'physformer':
        base = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(300, 64, 64), patches=(4, 4, 4),
            in_channels=1, theta=0.2)
        return PhysFormerWrapper(base).to(device)
    raise ValueError(model_type)


def infer_300(model, vid_300):
    x = torch.from_numpy(vid_300).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.cpu().numpy().flatten().astype(np.float64)


# ============================================================
# 2. CNA 실험 정의 (단일 split, ckpt = .pth 하나씩)
# ============================================================
DATA_CNA = {
    'with_regis':    '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_len300',
    'without_regis': '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped_len300',
}

# (라벨, ckpt.pth, 모델종류, 데이터키)
EXPERIMENTS_CNA = [
    ('PhysNet|without_regis|ncor+kldiv',
     '/home/son/orientation-uv-rppg/data/parameter_save/300franewithoutregis/klloss+ncorshift/0.0001_epoch:79_MAE: 10.00.pth',
     'physnet', 'without_regis'),

    ('PhysFormer|without_regis|ncor',
     '/home/son/orientation-uv-rppg/data/parameter_save/physformer/300franewithoutregis_13out_and_znorm/bandpassbeforelosscompute_ncorshift/0.0001_epoch:80_MAE: 10.10.pth',
     'physformer', 'without_regis'),

    ('PhysFormer|without_regis|ncor+kldiv',
     '/home/son/orientation-uv-rppg/data/parameter_save/physformer/300franewithoutregis_13out_and_znorm/ncorshift+kldiv/0.0001_epoch:61_MAE: 13.14.pth',
     'physformer', 'without_regis'),

    ('PhysNet|with_regis|ncor+kldiv',
     '/home/son/orientation-uv-rppg/data/parameter_save/300franewithregis_13out_and_znorm/bandpassbeforelosscompute_physformerloss/physent_data_wo_ol_lr:0.00030000000000000003_epoch:80_MAE: 13.70.pth',
     'physnet', 'with_regis'),

    ('PhysFormer|with_regis|ncor',
     '/home/son/orientation-uv-rppg/data/parameter_save/physformer/300franewithregis_13out_and_znorm/bandpassbeforelosscompute_ncorshift/0.0001_epoch:80_MAE: 12.66.pth',
     'physformer', 'with_regis'),

    # ── ncor 재학습 끝나면 아래에 추가 ──
    # ('PhysNet|without_regis|ncor', '<ckpt.pth>', 'physnet', 'without_regis'),
    # ('PhysNet|with_regis|ncor',    '<ckpt.pth>', 'physnet', 'with_regis'),
    # ('PhysFormer|with_regis|ncor+kldiv', '<ckpt.pth>', 'physformer', 'with_regis'),
]

VALID_START = 12   # all_files[12:]가 valid


# ============================================================
# 3. 실험 하나 평가 (valid 파일 전체의 윈도우별 HR/RMSSD)
# ============================================================
def eval_experiment(model_type, ckpt_path, data_dir):
    model = build_model(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    valid_files = all_files[VALID_START:]

    results = []
    for fname in valid_files:
        with h5py.File(os.path.join(data_dir, fname), 'r') as f:
            video  = f['video'][:]    # (N,300,64,64)
            gt_ppg = f['gt_ppg'][:]   # (N,300)
        for i in range(video.shape[0]):   # 각 10초 윈도우 독립
            rppg_300 = infer_300(model, video[i])
            gt_300   = gt_ppg[i].astype(np.float64)
            r_pred = compute_hr_rmssd(rppg_300, fps=30, use_bandpass=True)   # rPPG 필터
            r_gt   = compute_hr_rmssd(gt_300,   fps=30, use_bandpass=False)  # GT 이미 필터됨
            results.append({'pred': r_pred, 'gt': r_gt})

    del model
    torch.cuda.empty_cache()
    return results, valid_files


# ============================================================
# 4. 페어드 메트릭
# ============================================================
def paired_metrics(results, key):
    pairs = [(r['pred'][key], r['gt'][key]) for r in results
             if r['pred'] and r['gt'] and key in r['pred'] and key in r['gt']
             and np.isfinite(r['pred'][key]) and np.isfinite(r['gt'][key])]
    if len(pairs) < 2:
        return None
    pred = np.array([p for p,_ in pairs]); gt = np.array([g for _,g in pairs])
    err = pred - gt
    return {'MAE': np.mean(np.abs(err)), 'STD': np.std(err, ddof=1),
            'RMSE': np.sqrt(np.mean(err**2)),
            'R': pearsonr(pred,gt)[0] if np.std(pred)>0 and np.std(gt)>0 else np.nan,
            'pred_mean': pred.mean(), 'gt_mean': gt.mean(), 'n': len(pairs)}


# ============================================================
# 5. 메인
# ============================================================
METRICS = [('HR', 'mean_hr'), ('RMSSD', 'rmssd')]
summary = {}

for label, ckpt, model_type, data_key in EXPERIMENTS_CNA:
    if not os.path.exists(ckpt):
        print(f"[skip] {label}: ckpt 없음\n  {ckpt}")
        continue
    print(f"\n{'='*70}\n[{label}]  model={model_type} data={data_key}\n{'='*70}")
    try:
        results, vfiles = eval_experiment(model_type, ckpt, DATA_CNA[data_key])
    except Exception as ex:
        print(f"  [error] {ex}")
        continue

    total = len(results)
    vp = sum(1 for r in results if r['pred'] is not None)
    vg = sum(1 for r in results if r['gt']   is not None)
    summary[label] = {key: paired_metrics(results, key) for _, key in METRICS}
    print(f"  valid 파일 {len(vfiles)}개, 윈도우 {total} | "
          f"rPPG valid {vp} ({100*vp/max(total,1):.0f}%) | GT valid {vg}")


# ============================================================
# 6. 비교 테이블
# ============================================================
print("\n\n" + "="*84)
print("CNA 랩 데이터 | 실험별 HR / RMSSD 비교 (10초 윈도우, valid=all_files[12:])")
print("="*84)
print(f"{'Experiment':<36}{'HR_MAE':<9}{'HR_R':<8}"
      f"{'RMSSD_MAE':<11}{'RMSSD_R':<9}{'n':<6}")
print("-"*84)
for label, _, _, _ in EXPERIMENTS_CNA:
    if label not in summary:
        continue
    hr, rm = summary[label]['mean_hr'], summary[label]['rmssd']
    row = f"{label:<36}"
    row += f"{hr['MAE']:<9.2f}{hr['R']:<8.3f}" if hr else f"{'N/A':<9}{'N/A':<8}"
    row += f"{rm['MAE']:<11.2f}{rm['R']:<9.3f}" if rm else f"{'N/A':<11}{'N/A':<9}"
    row += f"{hr['n'] if hr else 0:<6}"
    print(row)
print("="*84)

# 상세 (bias 확인)
print("\n[상세]")
for label, _, _, _ in EXPERIMENTS_CNA:
    if label not in summary:
        continue
    print(f"\n{label}")
    for mlabel, key in METRICS:
        m = summary[label][key]
        if m:
            unit = 'BPM' if key == 'mean_hr' else 'ms'
            print(f"  {mlabel:<6} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R={m['R']:.3f}"
                  f" | pred={m['pred_mean']:.1f} gt={m['gt_mean']:.1f} {unit} (n={m['n']})")
        else:
            print(f"  {mlabel:<6} N/A")