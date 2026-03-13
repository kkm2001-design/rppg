import os
import torch
import torch.nn as nn 
import numpy as np

class NegCorLoss(nn.Module) :
    def __init__(self) :
        super().__init__()
        
    def forward(self, preds, targets) :
        preds_norm = preds - torch.mean(preds, dim = -1, keepdim = True)
        targets_norm = targets - torch.mean(targets, dim = -1, keepdim = True)

        correlation =  torch.sum(preds_norm * targets_norm, dim = -1) / (torch.sqrt(torch.sum(preds_norm ** 2, dim=-1)) * torch.sqrt(torch.sum(targets_norm ** 2, dim=-1)) + 1e-8 )
        return  torch.mean(1 - correlation)

