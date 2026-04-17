# tuning/trainer.py
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import os

from utils.metrics import MetricsCalculator
from utils.losses import CombinedLoss
from .config import Config

class ModelTrainer:
    def __init__(self, model, model_name, train_loader, val_loader, test_loader,
                 lr, batch_size, loss_name, optimizer_name, dropout):
        
        self.model = model.to(Config.DEVICE)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = self._get_loss(loss_name)
        self.optimizer = self._get_optimizer(optimizer_name, lr)
        
        self.metrics = MetricsCalculator()
        self.sigmoid = nn.Sigmoid()
        
        self.hyperparams = {'lr': lr, 'batch_size': batch_size, 'loss': loss_name, 'optimizer': optimizer_name, 'dropout': dropout}
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.epoch_metrics = []
    
    def _get_loss(self, loss_name):
        if loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'bce_dice':
            return CombinedLoss()
        return CombinedLoss()
    
    def _get_optimizer(self, optimizer_name, lr):
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for images, masks in tqdm(self.train_loader, desc="Training", leave=False):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                outputs_sigmoid = self.sigmoid(outputs)
                for i in range(images.size(0)):
                    total_dice += self.metrics.dice_score(outputs_sigmoid[i], masks[i]).item()
        return total_loss / len(self.val_loader), total_dice / len(self.val_loader.dataset)
    
    def test(self):
        self.model.eval()
        metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = self.model(images)
                outputs_sigmoid = self.sigmoid(outputs)
                for i in range(images.size(0)):
                    pred = (outputs_sigmoid[i] > 0.5).float()
                    target = masks[i]
                    intersection = (pred * target).sum()
                    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
                    metrics['dice'] += dice.item()
                    union = pred.sum() + target.sum() - intersection
                    metrics['iou'] += (intersection + 1e-6) / (union + 1e-6).item()
                    correct = (pred == target).sum()
                    metrics['accuracy'] += (correct / target.numel()).item()
        n = len(self.test_loader.dataset)
        for k in metrics:
            metrics[k] /= n
        return metrics
    
    def train(self, save_epoch_log=True, model_name_suffix=""):
        for epoch in range(Config.FINAL_EPOCHS):
            train_loss = self.train_epoch()
            val_loss, val_dice = self.validate()
            self.epoch_metrics.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_dice': val_dice})
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    break
        self.model.load_state_dict(self.best_model_state)
        test_metrics = self.test()
        return {
            'train_loss': self.epoch_metrics[-1]['train_loss'],
            'val_loss': self.best_val_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_dice': test_metrics['dice'],
            'test_iou': test_metrics['iou'],
            **self.hyperparams,
            'epochs_trained': len(self.epoch_metrics)
        }