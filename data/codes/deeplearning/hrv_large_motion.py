import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
# 0. 필터 + HR/RMSSD 추출 (10초 윈도우, robust)
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
    """10초 윈도우용. HR + RMSSD만. robust filtering 포함.
    Returns dict or None."""
    try:
        if use_bandpass:
            sig = bandpass_fft_soft(sig, fs=fps)
        clean = nk.ppg_clean(sig.copy(), sampling_rate=fps, method='elgendi')
        peaks = nk.ppg_findpeaks(clean, sampling_rate=fps)["PPG_Peaks"]
        if len(peaks) < min_beats:
            return None

        ibi_ms = np.diff(peaks) / fps * 1000
        # 생리학적 범위 (40~200 BPM)
        ibi_ms = ibi_ms[(ibi_ms > 300) & (ibi_ms < 1500)]
        if len(ibi_ms) < 2:
            return None
        # 인접 IBI가 median에서 ±50% 벗어나면 artifact로 제외
        med = np.median(ibi_ms)
        ibi_ms = ibi_ms[(ibi_ms > 0.5*med) & (ibi_ms < 1.5*med)]
        if len(ibi_ms) < min_beats - 1:
            return None

        diff = np.diff(ibi_ms)
        return {
            'mean_hr':   60000.0 / np.mean(ibi_ms),
            'rmssd':     np.sqrt(np.mean(diff**2)),
            'num_beats': len(ibi_ms) + 1,
        }
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
    """vid_300: (300,64,64) np → rppg (300,) np"""
    x = torch.from_numpy(vid_300).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return out.cpu().numpy().flatten().astype(np.float64)


# ============================================================
# 2. 실험 정의
# ============================================================
BASE = '/home/son/orientation-uv-rppg/data/parameter_save/nirpdriving'
DATA = {
    'cropped':  '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/driving_cropped_dl_data',
    'withregis':'/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_driving_withoutoutlier_len300_gtsubjectznorm',
}

# (라벨, ckpt폴더, 모델종류, 데이터키)
EXPERIMENTS = [
    ('PhysNet|cropped|ncorshift',
     f'{BASE}/LOSO_driving_cropped_ncorshift',                  'physnet',    'cropped'),
    ('PhysNet|withregis|ncorshift',
     f'{BASE}/LOSO_driving_withregis_ncorshift',                'physnet',    'withregis'),
    ('PhysNet|cropped|ncorshift+kldiv',
     f'{BASE}/physnet/LOSO_driving_cropped_ncorshift+kldiv',    'physnet',    'cropped'),
    ('PhysFormer|cropped|ncorshift',
     f'{BASE}/physformer/LOSO_driving_cropped_ncorshift',       'physformer', 'cropped'),
    ('PhysFormer|withregis|ncorshift',
     f'{BASE}/physformer/LOSO_driving_withregis_ncorshift',     'physformer', 'withregis'),
]


# ============================================================
# 3. fold 평가: 윈도우별 HR/RMSSD 페어 수집 (concat 없음)
# ============================================================
def get_subject_files(data_dir):
    return sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])


def eval_fold(model_type, ckpt_path, data_dir, subj_file):
    model = build_model(model_type)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    with h5py.File(os.path.join(data_dir, subj_file), 'r') as f:
        video  = f['video'][:]    # (N,300,64,64)
        gt_ppg = f['gt_ppg'][:]   # (N,300)
    N = video.shape[0]

    fold_results = []
    for i in range(N):              # 각 300프레임(10초) 윈도우 독립 처리
        rppg_300 = infer_300(model, video[i])
        gt_300   = gt_ppg[i].astype(np.float64)

        r_pred = compute_hr_rmssd(rppg_300, fps=30, use_bandpass=True)   # rPPG 필터
        r_gt   = compute_hr_rmssd(gt_300,   fps=30, use_bandpass=False)  # GT 이미 필터됨
        fold_results.append({'pred': r_pred, 'gt': r_gt})

    del model
    torch.cuda.empty_cache()
    return fold_results


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
# 5. 메인 — 전체 실험 순회
# ============================================================
METRICS = [('HR', 'mean_hr'), ('RMSSD', 'rmssd')]
summary = {}

for label, ckpt_dir, model_type, data_key in EXPERIMENTS:
    data_dir = DATA[data_key]
    subj_files = get_subject_files(data_dir)
    print(f"\n{'='*70}\n[{label}]  model={model_type} data={data_key} "
          f"subjects={len(subj_files)}\n{'='*70}")

    all_results, total_win, valid_pred, valid_gt = [], 0, 0, 0
    for fold_idx, subj_file in enumerate(tqdm(subj_files, desc="folds")):
        ckpt = os.path.join(ckpt_dir, f'fold_{fold_idx}', 'best_model.pth')
        if not os.path.exists(ckpt):
            print(f"  [skip] fold_{fold_idx}: ckpt 없음")
            continue
        try:
            fr = eval_fold(model_type, ckpt, data_dir, subj_file)
            all_results.extend(fr)
            total_win += len(fr)
            valid_pred += sum(1 for r in fr if r['pred'] is not None)
            valid_gt   += sum(1 for r in fr if r['gt']   is not None)
        except Exception as ex:
            print(f"  [error] fold_{fold_idx}: {ex}")

    summary[label] = {
        'metrics': {key: paired_metrics(all_results, key) for _, key in METRICS},
        'total_win': total_win, 'valid_pred': valid_pred, 'valid_gt': valid_gt,
    }
    print(f"  윈도우 {total_win} | rPPG valid {valid_pred} "
          f"({100*valid_pred/max(total_win,1):.0f}%) | GT valid {valid_gt}")


# ============================================================
# 6. 최종 비교 테이블
# ============================================================
print("\n\n" + "="*88)
print("실험별 HR / RMSSD 비교 (10초 윈도우, LOSO pooled, Shaffer&Ginsberg 2017 근거)")
print("="*88)
print(f"{'Experiment':<34}{'HR_MAE':<9}{'HR_R':<8}"
      f"{'RMSSD_MAE':<11}{'RMSSD_R':<9}{'n':<6}")
print("-"*88)
for label, _, _, _ in EXPERIMENTS:
    if label not in summary:
        continue
    m = summary[label]['metrics']
    row = f"{label:<34}"
    hr, rm = m['mean_hr'], m['rmssd']
    row += f"{hr['MAE']:<9.2f}{hr['R']:<8.3f}" if hr else f"{'N/A':<9}{'N/A':<8}"
    row += f"{rm['MAE']:<11.2f}{rm['R']:<9.3f}" if rm else f"{'N/A':<11}{'N/A':<9}"
    row += f"{hr['n'] if hr else 0:<6}"
    print(row)
print("="*88)

# 상세 (bias 확인용 pred_mean/gt_mean 포함)
print("\n[상세]")
for label, _, _, _ in EXPERIMENTS:
    if label not in summary:
        continue
    print(f"\n{label}")
    for mlabel, key in METRICS:
        m = summary[label]['metrics'][key]
        if m:
            unit = 'BPM' if key == 'mean_hr' else 'ms'
            print(f"  {mlabel:<6} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} "
                  f"R={m['R']:.3f} | pred={m['pred_mean']:.1f} gt={m['gt_mean']:.1f} {unit} "
                  f"(n={m['n']})")
        else:
            print(f"  {mlabel:<6} N/A")