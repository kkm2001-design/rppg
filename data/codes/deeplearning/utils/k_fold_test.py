import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils_sig import calculate_batch_mae
from utils.Loss import *
from utils.train import *





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
    pic_dir     = './pics',
    tag         = 'exp',
    k           = None,          # None → LOSO
):
    os.makedirs(pic_dir,   exist_ok=True)


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
        print(f'  Val files : {[os.path.basename(f) for f in val_files]}')
        print(f"{'='*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model     = model_fn().to(device)
        criterion = Neg_Pearson().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

        train_losses, val_losses = [], []
        train_maes,   val_maes   = [], []
        best_val_loss   = float('inf')
        best_val_mae    = float('inf')   # ← best loss 시점의 MAE
        best_min_mae    = float('inf')   # ← 전체 에폭 중 최소 MAE
        best_min_mae_epoch = 0
        best_epoch      = 0

        pbar = tqdm(range(1, num_epochs + 1), desc=f'Fold {fold_idx+1}', ncols=140)
        for epoch in pbar:
            train_metrics = train_one_epoch(model, train_loader, criterion, device, epoch)
            val_metrics = validate(model, val_loader, criterion, device, epoch)

            #scheduler.step()

            # 전체 에폭 중 최소 MAE 추적
            if vl_mae < best_min_mae:
                best_min_mae       = vl_mae
                best_min_mae_epoch = epoch

            pbar.set_postfix({
                'tr_loss': f'{tr_loss:.4f}',
                'tr_mae' : f'{tr_mae:.2f}',
                'vl_loss': f'{vl_loss:.4f}',
                'vl_mae' : f'{vl_mae:.2f}',
                #'lr'     : f'{scheduler.get_last_lr()[0]:.2e}'
            }, refresh=False)

            # best val loss 갱신 시 → MAE도 함께 저장
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                best_val_mae  = vl_mae   # ← best loss 시점 MAE 기록
                param_dir = f'/home/son/orientation-uv-rppg/data/parameter_save/nirpdriving{model_select.__name__}/{val_files[0]}'  ################################ best parameter 저장 경로
                os.makedirs(param_dir, exist_ok=True)
                # 이전 best 파일 삭제
                if best_epoch > 0:
                    old_ckpt = os.path.join(param_dir, f"{tag}_fold{fold_idx+1}_best_epoch{best_epoch}.pth")
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

                best_epoch = epoch
                ckpt_path = os.path.join(param_dir, f"{tag}_fold{fold_idx+1}_best_epoch{best_epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)

        # loss curve 저장
        curve_path = os.path.join(pic_dir, f"{tag}_fold{fold_idx+1}_loss_curve_batch{batch_size}_lr{lr}_epoch{best_epoch}.png")
        save_loss_curve(
            train_losses, val_losses,
            save_path=curve_path,
            title=f"{tag} | Fold {fold_idx+1} | Val: {val_subjects}"
        )

        print(f"  Best Val Loss  : {best_val_loss:.4f}  (epoch {best_epoch})  →  MAE @ best loss : {best_val_mae:.2f}")
        print(f"  Best Val MAE   : {best_min_mae:.2f}  (epoch {best_min_mae_epoch})")
        print(f"{'─'*60}")

        all_fold_results.append({
            'fold'              : fold_idx + 1,
            'val_subjects'      : val_subjects,
            'best_val_loss'     : best_val_loss,
            'best_epoch'        : best_epoch,
            'mae_at_best_loss'  : best_val_mae,   # ← best loss 시점 MAE
            'best_min_mae'      : best_min_mae,   # ← 전 에폭 최소 MAE
            'best_min_mae_epoch': best_min_mae_epoch,
        })

    # 전체 요약
    print(f"\n{'='*60}")
    print("K-Fold Summary")
    print(f"{'='*60}")
    for r in all_fold_results:
        print(
            f"  Fold {r['fold']} | Val: {r['val_subjects']} | "
            f"Best Loss: {r['best_val_loss']:.4f} (ep{r['best_epoch']}) | "
            f"MAE@BestLoss: {r['mae_at_best_loss']:.2f} | "
            f"MinMAE: {r['best_min_mae']:.2f} (ep{r['best_min_mae_epoch']})"
        )

    avg_loss         = np.mean([r['best_val_loss']    for r in all_fold_results])
    avg_mae_at_loss  = np.mean([r['mae_at_best_loss'] for r in all_fold_results])
    avg_min_mae      = np.mean([r['best_min_mae']     for r in all_fold_results])

    print(f"\n  Mean Best Val Loss    : {avg_loss:.4f}")
    print(f"  Mean MAE @ Best Loss  : {avg_mae_at_loss:.2f}")
    print(f"  Mean Min MAE          : {avg_min_mae:.2f}")

    return all_fold_results