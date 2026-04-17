# models/resunet.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class ResUNet(nn.Module):
    """Model 4: U-Net + ResNet (Residual blocks)"""
    def __init__(self, in_channels=3, out_channels=1, dropout=0.3):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock(64, dropout)
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock(128, dropout)
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True)
        )
        self.res3 = ResidualBlock(256, dropout)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
            
        )
    
    def forward(self, x):
        x = self.enc1(x)
        x = self.res1(x)
        x = self.enc2(x)
        x = self.res2(x)
        x = self.enc3(x)
        x = self.res3(x)
        return self.decoder(x)