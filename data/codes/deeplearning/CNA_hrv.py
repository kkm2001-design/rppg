import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import signal as scisig
from scipy.stats import pearsonr
from scipy.fft import fft
import neurokit2 as nk

import sys
sys.path.append('/home/son/orientation-uv-rppg/data/codes/deeplearning/utils')
from models import (PhysNet_padding_Encoder_Decoder_MAX,
                    ViT_ST_ST_Compact3_TDC_gra_sharp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 필터 + HR/RMSSD (10초 윈도우)
# ============================================================
def bandpass_fft_soft(sig, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    is_numpy = isinstance(sig, np.ndarray)
    if is_numpy:
        sig = torch.tensor(sig, dtype=torch.float32)
    N = sig.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(sig.device)
    fft_signal = torch.fft.rfft(sig)
    mask = torch.sigmoid(steepness * (freqs - low_hz)) * \
           torch.sigmoid(steepness * (high_hz - freqs))
    filtered = torch.fft.irfft(fft_signal * mask, n=N)
    return filtered.cpu().numpy() if is_numpy else filtered


def hr_fft(sig, fs=30, harmonics_removal=True, lowcut=0.66, highcut=2.5,
           nfft=4096, harm_ratio_thresh=0.5):
    """
    Zero-padding FFT 기반 HR 추정.
    harmonics_removal: hr1이 hr2의 2배에 가깝고 hr2가 의미 있는 magnitude를 가지면
                       hr2(fundamental) 채택.
    """
    sig = np.asarray(sig).reshape(-1)
    N = sig.shape[0]
    sig = sig * scisig.windows.hann(N)

    sig_f = np.abs(fft(sig, n=nfft))
    low_idx  = int(np.round(lowcut  / fs * nfft))
    high_idx = int(np.round(highcut / fs * nfft))

    sig_f_masked = sig_f.copy()
    sig_f_masked[:low_idx]  = 0
    sig_f_masked[high_idx:] = 0

    peak_idx, _ = scisig.find_peaks(sig_f_masked)
    if len(peak_idx) == 0:
        return np.nan
    if len(peak_idx) == 1:
        return peak_idx[0] / nfft * fs * 60

    sort_idx = np.argsort(sig_f_masked[peak_idx])[::-1]
    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]
    hr1 = peak_idx1 / nfft * fs * 60
    hr2 = peak_idx2 / nfft * fs * 60
    mag1 = sig_f_masked[peak_idx1]
    mag2 = sig_f_masked[peak_idx2]

    if harmonics_removal:
        is_harmonic = np.abs(hr1 - 2 * hr2) < 10
        mag_significant = (mag2 / mag1) >= harm_ratio_thresh
        if is_harmonic and mag_significant:
            return hr2
    return hr1


def compute_rmssd(sig, fps=30, use_bandpass=True, min_beats=3):
    """peak detection 기반 RMSSD. 필터/outlier 제거 없이 raw 그대로."""
    try:
        if use_bandpass:
            sig = bandpass_fft_soft(sig, fs=fps)
        clean = nk.ppg_clean(sig.copy(), sampling_rate=fps, method='elgendi')
        peaks = nk.ppg_findpeaks(clean, sampling_rate=fps)["PPG_Peaks"]
        if len(peaks) < min_beats:    # 최소 3개만 있으면 OK (IBI 2개 → diff 1개)
            return None
        ibi_ms = np.diff(peaks) / fps * 1000
        # IBI 범위/outlier 필터 없음 — false peak도 그대로 반영
        diff = np.diff(ibi_ms)
        return {'rmssd': np.sqrt(np.mean(diff**2)),
                'num_beats': len(peaks)}
    except Exception:
        return None


def compute_hr_rmssd(sig, fps=30, use_bandpass=True, min_beats=3):
    """
    HR : zero-padded FFT + harmonic removal (전 신호 dominant frequency)
    RMSSD : peak detection 기반 (raw, no IBI filtering)
    """
    sig_proc = bandpass_fft_soft(sig, fs=fps) if use_bandpass else sig

    hr_val = hr_fft(sig_proc, fs=fps)
    rmssd_info = compute_rmssd(sig_proc, fps=fps, use_bandpass=False, min_beats=min_beats)

    if np.isnan(hr_val) or rmssd_info is None:
        return None
    return {'mean_hr':   hr_val,
            'rmssd':     rmssd_info['rmssd'],
            'num_beats': rmssd_info['num_beats']}


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
# 2. CNA 실험 정의
# ============================================================
DATA_CNA = {
    'with_regis':    '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_70_withoutoutlier_len300',
    'without_regis': '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/dl_data_CNA_without_registration_cropped_len300',
}

EXPERIMENTS_CNA = [
    ('PhysNet|without_regis|ncor+kldiv',
     '/home/son/orientation-uv-rppg/data/parameter_save/300franewithoutregis/klloss+ncorshift/0.0001_epoch:80_MAE: 16.19.pth',
     'physnet', 'without_regis'),

    ('PhysFormer|without_regis|ncor',
     '/home/son/orientation-uv-rppg/data/parameter_save/physformer/300franewithoutregis_13out_and_znorm/bandpassbeforelosscompute_ncorshift/0.0001_epoch:80_MAE: 10.10.pth',
     'physformer', 'without_regis'),

    ('PhysFormer|without_regis|ncor+kldiv',
     '/home/son/orientation-uv-rppg/data/parameter_save/physformer/300franewithoutregis_13out_and_znorm/ncorshift+kldiv/0.0001_epoch:61_MAE: 13.14.pth',
     'physformer', 'without_regis'),

    ('PhysNet|with_regis|ncor+kldiv',
     '/home/son/orientation-uv-rppg/data/parameter_save/300franewithregis_13out_and_znorm/bandpassbeforelosscompute_physformerloss/physent_data_wo_ol_lr:0.00030000000000000003_epoch:70_MAE: 12.11.pth',
     'physnet', 'with_regis'),

    ('PhysFormer|with_regis|ncor',
     '/home/son/orientation-uv-rppg/data/parameter_save/physformer/300franewithregis_13out_and_znorm/bandpassbeforelosscompute_ncorshift/0.0001_epoch:80_MAE: 12.66.pth',
     'physformer', 'with_regis'),

    ('PhysNet|without_regis|ncor',
     '/home/son/orientation-uv-rppg/data/parameter_save/300franewithoutregis/klloss+ncorshift/ncor_without_shift/fold3/0.0001_epoch:57_MAE: 12.76.pth',
     'physnet', 'without_regis'),

    ('PhysNet|with_regis|ncor',
     '/home/son/orientation-uv-rppg/data/parameter_save/300franewithregis_13out_and_znorm/ncor_withoutshift/0.0001_epoch:80_MAE: 9.80.pth',
     'physnet', 'with_regis'),
]

VALID_START = 13


# ============================================================
# 3. 실험 하나 평가
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
            video  = f['video'][:]
            gt_ppg = f['gt_ppg'][:]
        for i in range(video.shape[0]):
            rppg_300 = infer_300(model, video[i])
            gt_300   = gt_ppg[i].astype(np.float64)
            r_pred = compute_hr_rmssd(rppg_300, fps=30, use_bandpass=True)
            r_gt   = compute_hr_rmssd(gt_300,   fps=30, use_bandpass=False)
            results.append({'pred': r_pred, 'gt': r_gt})

    del model
    torch.cuda.empty_cache()
    return results, valid_files


# ============================================================
# 4. 페어드 메트릭
# ============================================================
def paired_metrics(results, key, compute_mape=False):
    pairs = [(r['pred'][key], r['gt'][key]) for r in results
             if r['pred'] and r['gt'] and key in r['pred'] and key in r['gt']
             and np.isfinite(r['pred'][key]) and np.isfinite(r['gt'][key])]
    if len(pairs) < 2:
        return None
    pred = np.array([p for p,_ in pairs])
    gt   = np.array([g for _,g in pairs])
    abs_err = np.abs(pred - gt)
    err     = pred - gt

    m = {
        'MAE':       np.mean(abs_err),
        'MAE_std':   np.std(abs_err, ddof=1),
        'RMSE':      np.sqrt(np.mean(err**2)),
        'R':         pearsonr(pred, gt)[0] if np.std(pred)>0 and np.std(gt)>0 else np.nan,
        'pred_mean': pred.mean(),
        'gt_mean':   gt.mean(),
        'ratio':     pred.mean() / gt.mean() if gt.mean() != 0 else np.nan,
        'n':         len(pairs),
    }
    if compute_mape:
        valid = gt > 0
        if valid.sum() > 0:
            pct_err = abs_err[valid] / gt[valid] * 100
            m['MAPE']     = np.mean(pct_err)
            m['MAPE_std'] = np.std(pct_err, ddof=1)
        else:
            m['MAPE'] = m['MAPE_std'] = np.nan
    return m


# ============================================================
# 5. 메인
# ============================================================
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
    summary[label] = {
        'mean_hr': paired_metrics(results, 'mean_hr', compute_mape=False),
        'rmssd':   paired_metrics(results, 'rmssd',   compute_mape=True),
    }
    print(f"  valid 파일 {len(vfiles)}개, 윈도우 {total} | "
          f"rPPG valid {vp} ({100*vp/max(total,1):.0f}%) | GT valid {vg} ({100*vg/max(total,1):.0f}%)")


# ============================================================
# 6. 비교 테이블
# ============================================================
print("\n\n" + "="*110)
print("CNA 랩 데이터 | 실험별 HR / RMSSD 비교 (10초 윈도우, valid=all_files[12:])")
print("HR 계산: zero-padded FFT(nfft=4096) + harmonic removal")
print("RMSSD 계산: neurokit2 peak detection (필터 없음, raw)")
print("="*110)
print(f"{'Experiment':<38}{'HR MAE±σ (BPM)':<20}{'HR R':<8}"
      f"{'RMSSD MAE (ms)':<17}{'RMSSD MAPE (%)':<17}{'RMSSD R':<10}{'n':<5}")
print("-"*110)
for label, _, _, _ in EXPERIMENTS_CNA:
    if label not in summary:
        continue
    hr, rm = summary[label]['mean_hr'], summary[label]['rmssd']
    row = f"{label:<38}"
    if hr:
        row += f"{hr['MAE']:.2f} ± {hr['MAE_std']:.2f}   ".ljust(20)
        row += f"{hr['R']:<8.3f}"
    else:
        row += f"{'N/A':<20}{'N/A':<8}"
    if rm:
        row += f"{rm['MAE']:<17.2f}"
        row += f"{rm.get('MAPE', np.nan):<17.1f}"
        row += f"{rm['R']:<10.3f}"
    else:
        row += f"{'N/A':<17}{'N/A':<17}{'N/A':<10}"
    row += f"{hr['n'] if hr else 0:<5}"
    print(row)
print("="*110)


# ============================================================
# 7. PPT 표 기입용 한줄
# ============================================================
print("\n\n📋 PPT 표 기입용 (한 줄씩 복붙):")
print("-"*80)
for label, _, _, _ in EXPERIMENTS_CNA:
    if label not in summary:
        continue
    hr, rm = summary[label]['mean_hr'], summary[label]['rmssd']
    hr_str = f"{hr['MAE']:.1f} ± {hr['MAE_std']:.1f} BPM" if hr else "N/A"
    rm_str = f"MAPE {rm.get('MAPE', np.nan):.0f}%"        if rm else "N/A"
    print(f"  {label:<38} → HR: {hr_str:<22} | RMSSD: {rm_str}")
print("-"*80)


# ============================================================
# 8. 상세 (bias 확인)
# ============================================================
print("\n[상세 (bias 확인)]")
for label, _, _, _ in EXPERIMENTS_CNA:
    if label not in summary:
        continue
    print(f"\n{label}")
    hr, rm = summary[label]['mean_hr'], summary[label]['rmssd']
    if hr:
        print(f"  HR     MAE={hr['MAE']:.2f}±{hr['MAE_std']:.2f} RMSE={hr['RMSE']:.2f} "
              f"R={hr['R']:.3f} | pred={hr['pred_mean']:.1f} gt={hr['gt_mean']:.1f} BPM (n={hr['n']})")
    if rm:
        print(f"  RMSSD  MAE={rm['MAE']:.2f} RMSE={rm['RMSE']:.2f} R={rm['R']:.3f} "
              f"MAPE={rm.get('MAPE', np.nan):.1f}% "
              f"| pred={rm['pred_mean']:.1f} gt={rm['gt_mean']:.1f} ms "
              f"ratio={rm['ratio']:.2f}x (n={rm['n']})")