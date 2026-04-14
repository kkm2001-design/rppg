import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import pandas as pd
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import traceback
from scipy.signal import find_peaks,welch
from astropy.timeseries import LombScargle
import neurokit2 as nk
from scipy.interpolate import PchipInterpolator
from vmdpy import VMD  
from utils.registration_algorithm import MediaPipeLandmarkDetector

def z_norm(sig) :
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

def robust_norm(sig):
    q1, q3 = np.percentile(sig, [25, 75])
    iqr = q3 - q1
    return (sig - np.median(sig)) / (iqr + 1e-6)

def butter_bandpass_filter(data, lowcut = 0.66, highcut= 2.5 , fs = 30 , order=4):
    nyq = 0.5 * fs # 나이퀴스트 주파수
    low = lowcut / nyq
    high = highcut / nyq
    # 버터워스 필터 설계
    b, a = butter(order, [low, high], btype='band')
    # filtfilt를 사용하여 위상 지연 없이 필터링
    y = filtfilt(b, a, data)
    return y


def process_rppg_windows(avg_sig, sub_valid, window_size=300, max_gap_threshold=30, mode='normal', alp=800, k=3, lowcut=0.75, highcut=2.5, plot_g = False, plot_d = False,ref_ppg_sig = None):
    """
    rPPG 신호를 윈도우 단위로 자르고, 결손 구간 길이에 따라 보간 또는 폐기합니다.
    """
    fs = 30.0       # 영상의 FPS 

    valid_windows = {}
    discarded_indices = []

    if mode == 'normal' :
        sig_with_nan = avg_sig.copy()
        sig_with_nan[~sub_valid] = np.nan
        sig_series = pd.Series(sig_with_nan)

        n_samples = len(sig_series)
        n_windows = n_samples // window_size

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            window = sig_series.iloc[start_idx:end_idx].copy()
            
            is_nan = window.isna()
            groups = (is_nan != is_nan.shift()).cumsum()
            nan_groups = is_nan.groupby(groups)
            gap_lengths = nan_groups.size()[nan_groups.first()]
            max_gap = gap_lengths.max() if not gap_lengths.empty else 0
            
            if max_gap >= max_gap_threshold:
                discarded_indices.append(i)
            else:
                try:
                    if window.notna().sum() > 1:
                        interpolated_window = window.interpolate(method='pchip', limit_direction='both')
                        if interpolated_window.isna().any():
                            interpolated_window = interpolated_window.interpolate(method='linear', limit_direction='both')

                        detrend_sig = signal.detrend(interpolated_window)   
                        filter_sig = butter_bandpass_filter(detrend_sig, lowcut, highcut, fs, order=2)
                        f_sig = z_norm(filter_sig)

                        valid_windows[i] = f_sig
                                     

                    else:
                        discarded_indices.append(i)
                except Exception as e:
                    print(f'Window {i} Error : {e}')
                    discarded_indices.append(i)        
    
    print(f"총 {n_windows}개의 윈도우를 처리합니다...")
    print(f"처리 완료: 유효 윈도우 {len(valid_windows)}개, 폐기 윈도우 {len(discarded_indices)}개")
    return valid_windows, discarded_indices



def ppg_to_window(ppg, fail_index, window_size=900, lowcut=0.75, highcut=2.5) : 

    ppg_win_n = len(ppg) // window_size
    ppg_u = ppg.flatten().copy()

    ppg_windows = {}

    for i in range( ppg_win_n ) : 
        if i not in fail_index :
            start_ind = i * window_size
            end_ind = start_ind + window_size

            detrend_sig = signal.detrend( ppg_u[start_ind : end_ind] )
            ppg_f = butter_bandpass_filter( detrend_sig , lowcut, highcut, 30, order=2 )
            ppg_z = z_norm(ppg_f)

            ppg_windows[i] = ppg_z
        
        else : 
            continue

    return ppg_windows



def hr_fft(sig, fs, harmonics_removal=True, lowcut=0.66, highcut=2.5):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0]) # 양사이드 0처리
    sig_f = np.abs(fft(sig))
    
    low_idx = np.round(lowcut / fs * sig.shape[0]).astype('int') 
    high_idx = np.round(highcut / fs * sig.shape[0]).astype('int') 
    
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    
    if len(peak_idx) == 0 : 
        x_hr = np.arange(len(sig))/len(sig)*fs*60
        return  np.nan , sig_f_original, x_hr

    if len(peak_idx) == 1 :
        hr = (peak_idx[0] * fs) * 60 / sig.shape[0]
        x_hr = np.arange(len(sig))/len(sig)*fs*60
        return hr, sig_f_original, x_hr
        
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60

    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr




def get_hr(ppg, rppg, window_size=900, lowcut=0.75, highcut=2.5) : 
    
    ae_list = []
    rmse_list = []
    nan_list = []
    ppg_list = []
    rppg_list = []

    for i in list(ppg.keys()) :

        # hr_fft에 lowcut, highcut 인자 전달
        hr_ppg, psd_ppg, axis_ppg = hr_fft(ppg[i], fs=30, lowcut=lowcut, highcut=highcut)
        hr_rppg, psd_rppg, axis_rppg = hr_fft(rppg[i], fs=30, lowcut=lowcut, highcut=highcut)

        if np.isnan(hr_ppg) or np.isnan(hr_rppg) : 
            nan_list.append(i)
            continue 
        
        ae = abs(hr_ppg - hr_rppg)
        ae_list.append(ae)

        rmse = (hr_ppg - hr_rppg)**2
        rmse_list.append(rmse)

        ppg_list.append(hr_ppg)
        rppg_list.append(hr_rppg)

    mae = np.mean(ae_list)
    rmse = np.sqrt(np.mean(rmse_list))
    r = np.corrcoef(ppg_list, rppg_list)[0,1]

    return mae, nan_list , rmse , r




class get_hrv :
    
    def __init__(self, fs = 30 ) :
        self.fs = fs
        self.metrics = ['LF(n.u.)', 'HF(n.u.)', 'LF/HF', 'RF(Hz)']    
        self.results = {metric : {'ppg' : [], 'rppg' : []} for metric in self.metrics }
        self.failed_windows = []
    
    def _get_ibi(self, sig) : 
        
        try:
            # 1. 신호 전처리 (Elgendi Method)
            # copy()를 사용하여 원본 배열 보존
            ppg_clean = nk.ppg_clean(sig.copy(), sampling_rate=self.fs, method='elgendi')
            
            # 2. 피크 검출
            info = nk.ppg_findpeaks(ppg_clean, sampling_rate=self.fs)
            peak_loc = info["PPG_Peaks"]
            
            # 피크가 최소 2개는 있어야 간격(Interval) 계산 가능
            if len(peak_loc) < 2:
                return None
            
            # 3. IBI 계산 (초 단위)
            # (n번째 피크 - n-1번째 피크) / sampling_rate
            ibi = (peak_loc[1:] - peak_loc[:-1]) / self.fs
            return ibi
        except Exception:
            return None        
    
    def _get_freq_features(self, ibi) : 

        try :
            t_ibi = np.cumsum(ibi)

            # Lomb-Scargle Periodogram 계산
            # 주파수 범위: 0.01 ~ 0.51 Hz (VLF~HF 커버)
            ls = LombScargle(t_ibi, ibi)
            freq, power = ls.autopower(minimum_frequency=0.01, maximum_frequency=0.51)
            
            # LF (0.04 ~ 0.15 Hz) 구간 인덱싱
            idx_lf = np.where((freq >= 0.04) & (freq < 0.15))
            # HF (0.15 ~ 0.40 Hz) 구간 인덱싱
            idx_hf = np.where((freq >= 0.15) & (freq < 0.40))
            
            # Power Sum 계산
            power_lf = np.sum(power[idx_lf])
            power_hf = np.sum(power[idx_hf])
            
            total_power = power_lf + power_hf
            if total_power == 0: return None

            # Normalized Units (n.u.) 및 Ratio 계산
            lfnu = power_lf / total_power
            hfnu = power_hf / total_power
            ratio = power_lf / power_hf if power_hf != 0 else 0         

            return {
                'LF(n.u.)': lfnu, 
                'HF(n.u.)': hfnu, 
                'LF/HF': ratio
            }
        except Exception:
            return None               

    def _get_rf(self, sig):
        
        try:
            # Welch's Method로 PSD 계산 (전체 구간 사용)
            f, Pxx = welch(sig, fs=self.fs, nperseg=len(sig))
            
            # 호흡 주파수 대역 (0.1 ~ 0.5 Hz, 즉 분당 6~30회 호흡) 필터링
            resp_idx = np.where((f >= 0.1) & (f <= 0.5))
            f_resp = f[resp_idx]
            P_resp = Pxx[resp_idx]
            
            if len(P_resp) == 0: return np.nan
            
            # 가장 강한 주파수 성분(Peak)을 호흡수로 채택
            rf = f_resp[np.argmax(P_resp)]
            return rf
        except Exception:
            return np.nan

    def process_dataset(self, ppg_dict, rppg_dict) : 
        
        common_keys = sorted( list( set(ppg_dict.keys()) & set(rppg_dict.keys()) ))
        
        for i in common_keys :
            # ppg hrv 지표 산출
            ppg_w = np.array(ppg_dict[i]).flatten()
            ibi_ppg = self._get_ibi(ppg_w)
            if ibi_ppg is None:
                self.failed_windows.append(i)
                continue

            feats_ppg = self._get_freq_features(ibi_ppg)
            rf_ppg = self._get_rf(ppg_w) # 호흡수는 원본 신호에서 추출            

            #rppg hrv 지표 산출
            rppg_w = np.array(rppg_dict[i]).flatten()
            ibi_rppg = self._get_ibi(rppg_w)
            if ibi_rppg is None:
                self.failed_windows.append(i)
                continue 
            feats_rppg = self._get_freq_features(ibi_rppg)
            rf_rppg = self._get_rf(rppg_w) # 호흡수는 원본 신호에서 추출            


            if feats_rppg and feats_ppg and not np.isnan(rf_ppg) and not np.isnan(rf_rppg) :
                for k in ['LF(n.u.)', 'HF(n.u.)', 'LF/HF'] :
                    self.results[k]['ppg'].append(feats_ppg[k])
                    self.results[k]['rppg'].append(feats_rppg[k])
                
                self.results['RF(Hz)']['ppg'].append(rf_ppg)
                self.results['RF(Hz)']['rppg'].append(rf_rppg)
            else : 
                self.failed_windows.append(i)
                print('hrv 결과 append중 문제 발생') 
        
        print(f'hrv 분석완료 / 실패 window 개수 : { len( self.failed_windows ) } ')


    def print_table_metrics(self) :
        
        table = []
        for i in self.metrics :
            rppg_h = np.array(self.results[i]['rppg'])
            ppg_h = np.array(self.results[i]['ppg'])

            errors = rppg_h - ppg_h

            std = np.std(errors)
            rmse = np.sqrt(np.mean(errors**2))
            r = np.corrcoef(rppg_h, ppg_h)[0, 1]

            table.append( {
                'Metric' : i, 
                'STD' : round(std, 3), 
                'RMSE' : round(rmse, 3) , 
                'R' : round(r, 3)

            })
        
        df = pd.DataFrame(table)

        return df       



class get_sig(): #신호를 부위에 따라 얻는 class
    
    detector = MediaPipeLandmarkDetector()
    uv_numpy = detector.landmarks_uv.numpy()

    roi_r_ls = [101, 118, 117, 123, 187, 207, 206, 36]
    roi_l_ls = [330, 347, 346, 352, 411, 427, 426, 266]
    roi_ls_all = roi_r_ls + roi_l_ls

    roi_coords = (uv_numpy[roi_ls_all] * 64).astype(int)
    roi_rows = roi_coords[:,1]
    roi_cols = roi_coords[:,0]


    def __init__(self, frames):
        self.data = frames

    def get_rl_rois(self, rows, cols):
        T, H, W = self.data.shape
        f_sig = np.zeros(T)
        f_zcount = np.zeros(T)

        for r,c in zip(rows, cols) :
            sigs = self.data[:,r-1 : r+2,  c-1 : c+2]
            f_zcount += (sigs > 0).sum(axis = (1,2))
            f_sig += sigs.sum(axis = (1,2))
            
        f_sig = f_sig / (f_zcount + 1e-8)

        return f_sig


    def get_rl_roi_one(self, l_r, l_c, r_r, r_c) :

        roi_pixels_r = self.data[:, l_r-1 : l_r + 2, l_c-1 : l_c +2]
        nonzer_r = (roi_pixels_r > 0).sum(axis =(1,2))

        roi_pixels_l = self.data[:, r_r-1 : r_r + 2, r_c-1 : r_c +2]
        nonzer_l = (roi_pixels_l > 0).sum(axis =(1,2))

        roi_pixels = roi_pixels_r + roi_pixels_l
        nonzer_v = nonzer_l + nonzer_r

        sig_rl = roi_pixels.sum(axis = (1,2)) / (nonzer_v+1e-8)

        return sig_rl

    
    def get_rl_roi_left(self, l_r, l_c, r_r, r_c) :

        roi_pixels_l = self.data[:, l_r-1 : l_r + 2, l_c-1 : l_c +2]
        nonzer_l = (roi_pixels_l > 0).sum(axis =(1,2))

        sig_rl = roi_pixels_l.sum(axis = (1,2)) / (nonzer_l+1e-8)

        return sig_rl    

    def get_rl_allrois(self):
        T, H, W = self.data.shape
        
        # 1. 64x64 크기의 빈 검은색 캔버스(마스크) 생성
        mask = np.zeros((H, W), dtype=np.uint8)

        roi_r_coords = (self.uv_numpy[self.roi_r_ls] * 64).astype(np.int32)
        roi_l_coords = (self.uv_numpy[self.roi_l_ls] * 64).astype(np.int32)

        # 2. cv2.fillPoly에 맞게 차원 변경 (N, 1, 2) 형태의 int32 타입
        pts_r = roi_r_coords.reshape((-1, 1, 2)).astype(np.int32)
        pts_l = roi_l_coords.reshape((-1, 1, 2)).astype(np.int32)

        # 3. 캔버스에 오른쪽 볼, 왼쪽 볼 다각형을 그리고 안쪽을 1로 채움
        cv2.fillPoly(mask, [pts_r], 1)
        cv2.fillPoly(mask, [pts_l], 1)

        # 4. 마스크 값이 1인(다각형 내부) 픽셀들만 T 프레임 전체에서 한 번에 추출
        # 결과 shape은 (T, 다각형_내부_픽셀_총개수) 가 됩니다.
        roi_pixels = self.data[:, mask == 1]

        # 5. 추출된 픽셀 중 값이 0보다 큰(마스킹되지 않은) 진짜 피부 픽셀만 계산
        f_zcount = (roi_pixels > 0).sum(axis=1)
        f_sig = roi_pixels.sum(axis=1) / (f_zcount + 1e-8)

        return f_sig


# MAE 계산을 위한 헬퍼 함수
def calculate_batch_mae(preds, targets, fs=30):
    batch_size = preds.shape[0]
    mae_sum = 0.0
    valid_count = 0
    
    # GPU 텐서를 CPU 넘파이 배열로 변환
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    for i in range(batch_size):
        pred_sig = preds_np[i]
        gt_sig = targets_np[i]
        
        # 1. Bandpass Filter 적용 (노이즈 제거)
        pred_sig_filtered = butter_bandpass_filter(pred_sig, 0.66, 2.5, fs)

        
        # 2. FFT 기반 HR 추출
        pred_hr, _, _ = hr_fft(pred_sig_filtered, fs)
        gt_hr, _, _ = hr_fft(gt_sig, fs)
        
        # 신호가 너무 튀어서 peak를 못 찾은 경우(NaN) 예외 처리
        if not np.isnan(pred_hr) and not np.isnan(gt_hr):
            mae_sum += np.abs(pred_hr - gt_hr)
            valid_count += 1
            
    return mae_sum, valid_count

def bandpass_fft(signal, fs =30, low_hz=0.66, high_hz=2.5):

    N = signal.shape[-1]
    
    freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(signal.device)  # (N//2+1,)
    
    fft_signal = torch.fft.rfft(signal)  # (B, N//2+1) 복소수
    
    # 마스크: 통과 구간만 1
    mask = ((freqs >= low_hz) & (freqs <= high_hz)).float()  # (N//2+1,)
    
    fft_filtered = fft_signal * mask  # (B, N//2+1)
    filtered = torch.fft.irfft(fft_filtered, n=N)  # (B, T)
    return filtered


def bandpass_fft_soft(signal, fs=30, low_hz=0.66, high_hz=2.5, steepness=50.0):
    N = signal.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(signal.device)
    
    fft_signal = torch.fft.rfft(signal)
    
    # sigmoid로 soft boundary
    mask = torch.sigmoid(steepness * (freqs - low_hz)) * torch.sigmoid(steepness * (high_hz - freqs))
    
    fft_filtered = fft_signal * mask
    filtered = torch.fft.irfft(fft_filtered, n=N)
    return filtered