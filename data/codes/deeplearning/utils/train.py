import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils_sig import calculate_batch_mae


def validate(model, loader, criterion, device, fs=30):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total_valid_samples = 0

    with torch.no_grad():
        for inputs, targets in loader: 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # MAE 계산
            batch_mae_sum, valid_count = calculate_batch_mae(outputs, targets, fs)
            running_mae += batch_mae_sum
            total_valid_samples += valid_count
            
    avg_loss = running_loss / len(loader)
    avg_mae = running_mae / total_valid_samples if total_valid_samples > 0 else 0.0
    return avg_loss, avg_mae


def train_one_epoch(model, loader, criterion, optimizer, device, fs=30):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_valid_samples = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        #outputs = model(inputs)
        loss = criterion(outputs, targets)
        
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