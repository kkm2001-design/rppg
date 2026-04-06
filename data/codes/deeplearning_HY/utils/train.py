import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils_sig import calculate_batch_mae
from tqdm import tqdm 

def validate(model, loader, criterion, device, fs=30):
    model.eval()
    running_loss = 0.0
    running_denom = 0.0
    running_loss_time=0.0
    running_loss_CE=0.0
    running_loss_HR=0.0
            
    running_mae = 0.0
    total_valid_samples = 0
    main_pbar = tqdm(loader, desc="eval_segment")
    with torch.no_grad():
        #for inputs, targets, pos_pad, pad_mask  in main_pbar: 
        for inputs, targets  in main_pbar: 
            inputs, targets = inputs.to(device), targets.to(device)
            #pos_pad, pad_mask = pos_pad.to(device), pad_mask.to(device)
            #outputs = model(inputs,pad_mask)

            outputs,_,_,_ = model(inputs)

            #print(f'pred shape : {outputs.shape}')
            #print(f'pred mean  : {outputs.mean().item():.6f}')
            #print(f'pred std   : {proutputsed.std().item():.6f}')
            #print(f'pred max   : {outputs.max().item():.6f}')
            #print(f'pred min   : {outputs.min().item():.6f}')
            #print(f'gt mean    : {gt_batch.mean().item():.6f}')
            #print(f'gt std     : {gt_batch.std().item():.6f}')
            #print(f'pred std per sample: {outputs.std(dim=-1)}')

            loss,numer,denom = criterion(outputs, targets)
            #print(f"loss: {loss.item():.4f}, numer: {numer.item():.4f}, denom: {denom.item():.4f}")
            #print(f"pred std: {outputs.std(dim=-1).mean().item():.6f}")
            #loss,time,CE,HR = criterion(outputs, targets, 0)
            running_loss += loss.item()
            running_denom += denom.item()

            #running_loss_time += time.item()
            #running_loss_CE += CE.item()
            #running_loss_HR += HR.item()
            
            # MAE 계산
            batch_mae_sum, valid_count = calculate_batch_mae(outputs, targets, fs)
            running_mae += batch_mae_sum
            total_valid_samples += valid_count
            
    avg_loss = running_loss / len(loader)
    avg_denom = running_denom / len(loader)

    #avg_loss_time = running_loss_time / len(loader)
    #avg_loss_CE = running_loss_CE / len(loader)
    #avg_loss_HR = running_loss_HR / len(loader)

    avg_mae = running_mae / total_valid_samples if total_valid_samples > 0 else 0.0
    #return avg_loss, avg_mae, avg_loss_time,avg_loss_CE,avg_loss_HR
    return avg_loss, avg_mae, avg_denom


def train_one_epoch(model, loader, criterion, optimizer, device, fs=30):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_valid_samples = 0
    main_pbar = tqdm(loader, desc="train_segment")
    #for inputs, targets,pos_pad, pad_mask  in main_pbar:
    for inputs, targets in main_pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        #pos_pad, pad_mask = pos_pad.to(device), pad_mask.to(device)
        
        optimizer.zero_grad()
        #outputs = model(inputs,pad_mask)
        outputs,_,_,_ = model(inputs)
        loss,numer,denom = criterion(outputs, targets)
        #print(f"loss: {loss_val.item():.4f}, numer: {numer.item():.4f}, denom: {denom.item():.4f}")
        #print(f"pred std: {pred.std(dim=-1).mean().item():.6f}")
        #loss,_,_,_ = criterion(outputs, targets, 0)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # MAE 계산
        batch_mae_sum, valid_count = calculate_batch_mae(outputs, targets, fs)
        running_mae += batch_mae_sum
        total_valid_samples += valid_count

    avg_loss = running_loss / len(loader)
    avg_mae = running_mae / total_valid_samples if total_valid_samples > 0 else 0.0
    return avg_loss, avg_mae