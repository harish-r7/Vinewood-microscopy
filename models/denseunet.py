# models/denseunet.py
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.drop(out)
        return torch.cat([x, out], dim=1)

class DenseUNet(nn.Module):
    """Model 5: U-Net + DenseNet (Dense blocks)"""
    def __init__(self, in_channels=3, out_channels=1, dropout=0.3):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.dense1 = DenseBlock(64, 32, dropout)
        self.dense2 = DenseBlock(96, 32, dropout)
        self.dense3 = DenseBlock(128, 32, dropout)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(160, 64, 2, stride=2),
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.Conv2d(16, out_channels, 1)
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.dense1(x)
        x = self.pool(x)
        x = self.dense2(x)
        x = self.pool(x)
        x = self.dense3(x)
        return self.decoder(x)