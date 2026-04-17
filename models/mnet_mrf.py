# models/mnet_mrf.py
import torch
import torch.nn as nn
from .mnet import MNet

class MRFBlock(nn.Module):
    """Markov Random Field-like block for multi-scale features"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.drop(x)
        x = self.relu(self.conv2(x))
        x = self.drop(x)
        x = self.conv3(x)
        return torch.sigmoid(x)

class MNetMRF(nn.Module):
    """Model 7: M-Net + MRF (Markov Random Field)"""
    def __init__(self, in_channels=3, out_channels=1, dropout=0.3):
        super().__init__()
        self.mnet = MNet(in_channels, out_channels, dropout)
        self.mrf = MRFBlock(out_channels, out_channels, dropout)
    
    def forward(self, x):
        x = self.mnet(x)
        x = self.mrf(x)
        return x