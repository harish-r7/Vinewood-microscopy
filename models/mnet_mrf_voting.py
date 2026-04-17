# models/mnet_mrf_voting.py
import torch
import torch.nn as nn
from .mnet_mrf import MNetMRF

class MNetMRFVoting(nn.Module):
    """Model 8: M-Net + MRF + Majority Voting (Ensemble)"""
    def __init__(self, in_channels=3, out_channels=1, dropout=0.3):
        super().__init__()
        self.model1 = MNetMRF(in_channels, out_channels, dropout)
        self.model2 = MNetMRF(in_channels, out_channels, dropout)
        self.model3 = MNetMRF(in_channels, out_channels, dropout)
    
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        # Majority voting via average
        return (out1 + out2 + out3) / 3