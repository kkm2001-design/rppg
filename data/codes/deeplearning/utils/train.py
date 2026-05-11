import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils_sig import calculate_batch_mae, z_norm, bandpass_fft, bandpass_fft_soft
from utils.Loss import *

def z_norm_per_sample(x):
    # x: (B, T)
    mean = x.mean(dim=-1, keepdim=True)   # (B, 1)
    std  = x.std(dim=-1, keepdim=True)    # (B, 1)
    return (x - mean) / (std + 1e-8)      # per-sample


def validate(model, loader, criterion, device, epoch, fs=30):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    running_ncor = 0.0
    running_kl   = 0.0
    running_ce   = 0.0
    total_valid_samples = 0

    a_start = 1   
    b_start = 0   
    exp_a   = 0.5   
    exp_b   = 2.0   

    is_hybrid     = isinstance(criterion, HybridLoss)
    is_negcor     = isinstance(criterion, Neg_Pearson) 
    is_physformer = isinstance(criterion, physformerLoss)
    is_ncors = isinstance(criterion ,NCorshift)

    shift_list_save = []
    ncor_fn = NCorshift().to(device)
    shift_list = None
    with torch.no_grad():
        for inputs, targets, hr in loader:
            inputs, targets, hr = inputs.to(device), targets.to(device), hr.to(device)
            outputs = model(inputs)
            if outputs.shape[-1] != targets.shape[-1] :
                outputs = outputs[:,:-1]
            outputs = bandpass_fft_soft(outputs)
            outputs = z_norm_per_sample(outputs)


            if is_hybrid:
                loss, loss_ncor, loss_mse = criterion(outputs, targets)
                running_ncor += loss_ncor.item()
                running_mse  += loss_mse.item()

            elif is_negcor:
                loss = criterion(outputs, targets)

            elif is_ncors :
                loss,  shift_list = criterion(outputs, targets, epoch)     
                shift_list_save.append(shift_list.detach().cpu().numpy())

            elif is_physformer:
                loss_kl, _, _ = criterion.cross_entropy_power_spectrum_DLDL_softmax2(
                    outputs, hr, Fs=30, std=1.0)
                
                ncorloss, shift_list = ncor_fn(outputs, targets,epoch)
                shift_list_save.append(shift_list.detach().cpu().numpy())

                if epoch <= 30:
                    a = a_start
                    b = b_start

                else:
                    # exp descend
                    a = a_start*math.pow(exp_a, epoch/100.0) * math.sqrt(2)
                    # exp ascend
                    b = math.pow(exp_b, epoch/100.0) / math.sqrt(2)

                loss = a* ncorloss + b*(loss_kl)          # running_loss에 넣을 total
                running_ncor += ncorloss.item()
                running_kl   += loss_kl.item()
       

            running_loss += loss.item()
            batch_mae_sum, valid_count = calculate_batch_mae(outputs, targets, fs, shift_list)
            running_mae         += batch_mae_sum
            total_valid_samples += valid_count

    avg_loss = running_loss / len(loader)
    avg_mae  = running_mae / total_valid_samples if total_valid_samples > 0 else 0.0

    metrics = {'loss': avg_loss, 'mae': avg_mae}

    if is_hybrid:
        metrics['ncor'] = running_ncor / len(loader)
        metrics['mse']  = running_mse  / len(loader)
    elif is_physformer:
        metrics['ncor'] = running_ncor / len(loader)
        metrics['kl']   = running_kl   / len(loader)
        metrics['shift_list'] = np.concatenate(shift_list_save)
    elif is_ncors :
        metrics['shift_list'] = np.concatenate(shift_list_save)

    return metrics



def train_one_epoch(model, loader, criterion, optimizer, device, epoch, fs=30):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    running_ncor = 0.0   
    running_mse  = 0.0   
    running_kl = 0.0
    running_ce = 0.0
    total_valid_samples = 0

    a_start = 1   
    b_start = 0   
    exp_a   = 0.5   
    exp_b   = 2.0   
    
    is_hybrid = isinstance(criterion, HybridLoss) 
    is_negcor     = isinstance(criterion, Neg_Pearson) 
    is_physformer = isinstance(criterion, physformerLoss)
    is_ncors = isinstance(criterion ,NCorshift)

    ncor_fn = NCorshift().to(device) 
    shift_list = None
    for inputs, targets, hr in loader:
        inputs, targets, hr = inputs.to(device), targets.to(device), hr.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = bandpass_fft_soft(outputs)
        outputs = z_norm_per_sample(outputs) ## 입력전 znorm physformer 논문기반
        if outputs.shape[-1] != targets.shape[-1] :
            outputs = outputs[:,:-1]
        
        if is_hybrid:
            loss, loss_ncor, loss_mse = criterion(outputs, targets)
            running_ncor += loss_ncor.item()
            running_mse  += loss_mse.item()

        elif is_negcor:
            loss = criterion(outputs, targets)

        elif is_ncors :
            loss, shift_list = criterion(outputs, targets, epoch)     

        elif is_physformer:
            loss_kl, _, _ = criterion.cross_entropy_power_spectrum_DLDL_softmax2(
                outputs, hr, Fs=30, std=1.0)
            
            ncorloss, shift_list = ncor_fn(outputs, targets, epoch)

            if epoch <= 30:
                a = a_start
                b = b_start
            else:
                # exp descend
                a = a_start*math.pow(exp_a, epoch/100.0) * math.sqrt(2)
                # exp ascend
                b = math.pow(exp_b, epoch/100.0) / math.sqrt(2)

            loss = a* ncorloss + b*(loss_kl)          # running_loss에 넣을 total
            running_ncor += ncorloss.item()
            running_kl   += loss_kl.item()


        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batch_mae_sum, valid_count = calculate_batch_mae(outputs, targets, fs, shift_list)
        running_mae += batch_mae_sum
        total_valid_samples += valid_count

    avg_loss = running_loss / len(loader)
    avg_mae = running_mae / total_valid_samples if total_valid_samples > 0 else 0.0
    
    metrics = {'loss': avg_loss, 'mae': avg_mae}
    if is_hybrid:
        metrics['ncor'] = running_ncor / len(loader)
        metrics['mse']  = running_mse  / len(loader)
    elif is_physformer:
        metrics['ncor'] = running_ncor / len(loader)
        metrics['kl']   = running_kl   / len(loader)
        
    return metrics