# utils/losses.py - Fixed for mixed precision
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred already has sigmoid applied, but for numerical stability
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use BCEWithLogitsLoss instead of BCELoss for mixed precision
        # This combines sigmoid + BCE internally for better numerical stability
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, pred, target):
        # pred is logits (no sigmoid applied yet)
        bce_loss = self.bce(pred, target)
        # Apply sigmoid for dice loss
        pred_sigmoid = self.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target)
        return bce_loss + dice_loss