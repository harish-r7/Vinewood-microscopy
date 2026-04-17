# tuning/helpers.py
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloaders
from utils.metrics import MetricsCalculator
from utils.losses import CombinedLoss
from .config import Config

def get_dataloaders_for_tuning(batch_size):
    return get_dataloaders(
        Config.DATA_PATH,
        batch_size,
        num_workers=Config.DATALOADER_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

def quick_train_eval(model_class, params, train_loader, val_loader):
    model = model_class(Config.IN_CHANNELS, Config.OUT_CHANNELS, dropout=params['dropout']).to(Config.DEVICE)
    
    if params['loss'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = CombinedLoss()
    
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
    
    sigmoid = nn.Sigmoid()
    metrics = MetricsCalculator()
    best_val_loss = float('inf')
    
    patience = 0
    for epoch in range(Config.TUNING_EPOCHS):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= Config.TUNING_EARLY_STOPPING_PATIENCE:
                break
    
    return best_val_loss
