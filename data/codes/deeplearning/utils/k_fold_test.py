import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils_sig import calculate_batch_mae
from utils.Loss import *



# ──────────────────────────────────────────────
# forward 헬퍼 (PhysNet은 tuple 반환, 나머지는 tensor)
# ──────────────────────────────────────────────
def forward(model, inputs):
    out = model(inputs)
    if isinstance(out, tuple):
        return out[0]
    return out


# ──────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, fs=30):
    model.train()
    running_loss, running_mae, total_valid = 0.0, 0.0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = forward(model, inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        mae_sum, cnt = calculate_batch_mae(outputs, targets, fs)
        running_mae += mae_sum
        total_valid += cnt

    avg_loss = running_loss / len(loader)
    avg_mae  = running_mae  / total_valid if total_valid > 0 else 0.0
    return avg_loss, avg_mae


def validate(model, loader, criterion, device, fs=30):
    model.eval()
    running_loss, running_mae, total_valid = 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = forward(model, inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            mae_sum, cnt = calculate_batch_mae(outputs, targets, fs)
            running_mae += mae_sum
            total_valid += cnt

    avg_loss = running_loss / len(loader)
    avg_mae  = running_mae  / total_valid if total_valid > 0 else 0.0
    return avg_loss, avg_mae


# ──────────────────────────────────────────────
# Loss curve 저장
# ──────────────────────────────────────────────
def save_loss_curve(train_losses, val_losses, save_path, title='Loss Curve'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='steelblue')
    plt.plot(epochs, val_losses,   label='Val Loss',   color='tomato')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Neg Pearson)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [saved] {save_path}")


# ──────────────────────────────────────────────
# K-Fold runner
# ──────────────────────────────────────────────
def run_kfold(
    model_fn,
    subject_file_map,
    Dataset_cls,
    norm_type,
    device,
    num_epochs  = 100,
    batch_size  = 8,
    lr          = 1e-4,
    pic_dir     = './pics',      # loss curve 사진 저장 경로
    param_dir   = './params',    # .pth 파라미터 저장 경로
    tag         = 'exp',
    k           = None,          # None → LOSO
):
    os.makedirs(pic_dir,   exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)

    subject_ids = list(subject_file_map.keys())

    # fold 구성
    if k is None:
        folds = [[sid] for sid in subject_ids]   # LOSO
    else:
        groups = np.array_split(subject_ids, k)
        folds  = [list(g) for g in groups]

    all_fold_results = []

    for fold_idx, val_subjects in enumerate(folds):
        train_subjects = [s for s in subject_ids if s not in val_subjects]

        train_files = [f for s in train_subjects for f in subject_file_map[s]]
        val_files   = [f for s in val_subjects   for f in subject_file_map[s]]

        train_ds = Dataset_cls(train_files, norm_type=norm_type)
        val_ds   = Dataset_cls(val_files,   norm_type=norm_type)

        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/{len(folds)}  |  Val: {val_subjects}")
        print(f"  Train subjects : {train_subjects}")
        print(f"  Train windows  : {len(train_ds)}  |  Val windows: {len(val_ds)}")
        print(f'  Val files : {val_ds}')
        print(f"{'='*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model     = model_fn().to(device)
        criterion = Neg_Pearson().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        train_maes,   val_maes   = [], []
        best_val_loss = float('inf')

        pbar = tqdm(range(1, num_epochs + 1), desc=f'Fold {fold_idx+1}', ncols=110)
        for epoch in pbar:
            tr_loss, tr_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_mae = validate(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_maes.append(tr_mae)
            val_maes.append(vl_mae)

            pbar.set_postfix({
                'tr_loss': f'{tr_loss:.4f}',
                'tr_mae' : f'{tr_mae:.2f}',
                'vl_loss': f'{vl_loss:.4f}',
                'vl_mae' : f'{vl_mae:.2f}',
            })

            # best model 저장 → param_dir
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                ckpt_path = os.path.join(param_dir, f"{tag}_fold{fold_idx+1}_best_epoch{epoch}.pth")
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)

        # loss curve 저장 → pic_dir
        curve_path = os.path.join(pic_dir, f"{tag}_fold{fold_idx+1}_loss_curve_batch{batch_size}_lr{lr}_epoch{best_epoch}.png")
        save_loss_curve(
            train_losses, val_losses,
            save_path = curve_path,
            title     = f"{tag} | Fold {fold_idx+1} | Val: {val_subjects}"
        )
        print(f"  Best Val Loss : {best_val_loss:.4f}  |  Final Val MAE : {val_maes[-1]:.2f}")
        print(f"{'─'*60}")

        all_fold_results.append({
            'fold'         : fold_idx + 1,
            'val_subjects' : val_subjects,
            'best_val_loss': best_val_loss,
            'final_val_mae': val_maes[-1],
        })

    # 전체 요약
    print(f"\n{'='*60}")
    print("K-Fold Summary")
    print(f"{'='*60}")
    for r in all_fold_results:
        print(f"  Fold {r['fold']} | Val: {r['val_subjects']} | "
              f"Best Val Loss: {r['best_val_loss']:.4f} | Final MAE: {r['final_val_mae']:.2f}")
    avg_loss = np.mean([r['best_val_loss'] for r in all_fold_results])
    avg_mae  = np.mean([r['final_val_mae']  for r in all_fold_results])
    print(f"\n  Mean Best Val Loss: {avg_loss:.4f} | Mean Final MAE: {avg_mae:.2f}")

    return all_fold_results