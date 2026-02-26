import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re
import string
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader):
        indices = batch['x'].to(device)
        labels  = batch['y'].to(device)
        # lengths = batch['lengths']           # we don't use packed yet
        
        optimizer.zero_grad()
        logits = model(indices)
        loss = loss_fn(logits, labels.unsqueeze(dim=1))
        
        loss.backward()
        # Optional but better to use gradient clipping for RNNs to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels.unsqueeze(dim=1)).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            indices = batch['x'].to(device)
            labels  = batch['y'].to(device)
            # lengths = batch['lengths']
            
            logits = model(indices)
            loss = loss_fn(logits, labels.unsqueeze(dim=1))
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels.unsqueeze(dim=1)).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total