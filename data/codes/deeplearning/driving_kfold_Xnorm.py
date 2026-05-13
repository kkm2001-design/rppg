import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils.utils_sig import *
from utils.Loss import *
from utils.train import *
from utils.dataloader import *
from utils.models import *

print('LOSO / driving_cropped_Xnorm / 300frame / NCorshift + kldiv loss / withoutregis/ physnet model')

data_dir = '/home/son/orientation-uv-rppg/video_link/CNA_members/KM_preprocess_data/driving_Xnorm_dl_data_f'
save_root = '/home/son/orientation-uv-rppg/data/parameter_save/nirpdriving/physnet/LOSO_driving_Xnorm_ncorshift'
shift_root = '/home/son/orientation-uv-rppg/data/shift_list_save/nirpdriving/physnet/LOSO_driving_Xnorm_ncorshift'

all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')])
n_subjects = len(all_files)
print(f"Total subjects: {n_subjects}")
for i, f in enumerate(all_files):
    print(f"  [{i}] {os.path.basename(f)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 


lr_ = 1e-4
num_epochs = 80
batch_size = 24
fs = 30

fold_results = []

for fold_idx in range(n_subjects):

    if fold_idx > 0:
        del model, optimizer, train_loader, test_loader
        import gc; gc.collect()
        torch.cuda.empty_cache()

    test_files  = [all_files[fold_idx]]
    train_files = [all_files[i] for i in range(n_subjects) if i != fold_idx]

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: test={os.path.basename(test_files[0])}, train: {len(train_files)}")

    fold_save_dir = os.path.join(save_root, f'fold_{fold_idx}')
    fold_shift_dir = os.path.join(shift_root, f'fold_{fold_idx}')
    os.makedirs(fold_save_dir, exist_ok=True)
    os.makedirs(fold_shift_dir, exist_ok=True)

    train_loader = DataLoader(rPPGDataset(train_files), batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(rPPGDataset(test_files),  batch_size=batch_size, shuffle=False, num_workers=4)

    model = PhysNet_padding_Encoder_Decoder_MAX().to(device) 

    criterion = NCorshift()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_)

    best_test_loss = float('inf')
    best_test_mae = float('inf')
    mae_at_best_test_loss = 0


    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1, fs)
        test_metrics  = validate(model, test_loader, criterion, device, epoch+1, fs)

        if isinstance(criterion, physformerLoss) :
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} NCOR:{train_metrics['ncor']:.4f}, kl:{train_metrics['kl']:.4f}, MAE: {train_metrics['mae']:.2f} | " 
                f"Val Loss: {test_metrics['loss']:.4f} NCOR:{test_metrics['ncor']:.4f}, kl:{test_metrics['kl']:.4f}, MAE: {test_metrics['mae']:.2f} | "
            )

        else :
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f} | " 
                f"Val Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.2f} | "
        )

        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            mae_at_best_test_loss = test_metrics['mae']
            torch.save(model.state_dict(), os.path.join(fold_save_dir, 'best_model.pth'))
            if 'shift_list' in test_metrics:
                np.save(os.path.join(fold_shift_dir, f'{epoch+1}_best_test_shifts.npy'), test_metrics['shift_list'])

        if test_metrics['mae'] < best_test_mae  :
            best_test_mae = test_metrics['mae']
            if 'shift_list' in test_metrics:
                np.save(os.path.join(fold_shift_dir, f'{epoch+1}_best_test_shifts.npy'), test_metrics['shift_list'])

    fold_results.append({
        'fold': fold_idx,
        'test_subject': os.path.basename(test_files[0]),
        'mabt' : mae_at_best_test_loss,
        'best_test_loss': best_test_loss,
        'best_test_mae': best_test_mae
    })

# 최종 요약
print(f"\n{'='*60}")
print("LOSO Results Summary")
print(f"{'='*60}")
losses, maes, bmaes = [], [], []
for r in fold_results:
    print(f"Fold {r['fold']:2d} | {r['test_subject']:40s} | Loss: {r['best_test_loss']:.4f} | mae_at_best_loss : {r['mabt'] :.2f} | bestMAE: {r['best_test_mae']:.2f}")
    losses.append(r['best_test_loss'])
    maes.append(r['mabt'])
    bmaes.append(r['best_test_mae'])

print(f"{'='*60}")
print(f"Mean Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
print(f"Mean MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f}")
print(f"Mean best_MAE:  {np.mean(bmaes):.2f} ± {np.std(bmaes):.2f}")

