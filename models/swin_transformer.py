# models/swin_transformer.py - Completely Fixed
import torch
import torch.nn as nn

class SimplifiedSwin(nn.Module):
    """Model 3: Simplified Swin Transformer U-Net - Fixed channel issues"""
    def __init__(self, in_channels=3, out_channels=1, dropout=0.3):
        super().__init__()
        
        # Encoder (downsampling)
        # Level 1: 256 -> 128
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        
        # Level 2: 128 -> 64
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Level 3: 64 -> 32
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Level 4: 32 -> 16
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck: 16 -> 16
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (upsampling)
        # Level 4: 16 -> 32
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),  # 512 (up) + 512 (skip) = 1024
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Level 3: 32 -> 64
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 256 (up) + 256 (skip) = 512
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Level 2: 64 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 (up) + 128 (skip) = 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Level 1: 128 -> 256
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 (up) + 64 (skip) = 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)      # 256x256, 64 channels
        p1 = self.pool1(e1)    # 128x128, 64 channels
        
        e2 = self.enc2(p1)     # 128x128, 128 channels
        p2 = self.pool2(e2)    # 64x64, 128 channels
        
        e3 = self.enc3(p2)     # 64x64, 256 channels
        p3 = self.pool3(e3)    # 32x32, 256 channels
        
        e4 = self.enc4(p3)     # 32x32, 512 channels
        p4 = self.pool4(e4)    # 16x16, 512 channels
        
        # Bottleneck
        b = self.bottleneck(p4)  # 16x16, 512 channels
        
        # Decoder with skip connections
        d4 = self.up4(b)          # 32x32, 512 channels
        d4 = torch.cat([d4, e4], dim=1)  # 512 + 512 = 1024 channels
        d4 = self.dec4(d4)        # 32x32, 512 channels
        
        d3 = self.up3(d4)         # 64x64, 256 channels
        d3 = torch.cat([d3, e3], dim=1)  # 256 + 256 = 512 channels
        d3 = self.dec3(d3)        # 64x64, 256 channels
        
        d2 = self.up2(d3)         # 128x128, 128 channels
        d2 = torch.cat([d2, e2], dim=1)  # 128 + 128 = 256 channels
        d2 = self.dec2(d2)        # 128x128, 128 channels
        
        d1 = self.up1(d2)         # 256x256, 64 channels
        d1 = torch.cat([d1, e1], dim=1)  # 64 + 64 = 128 channels
        d1 = self.dec1(d1)        # 256x256, 64 channels
        
        # Final output
        out = self.final(d1)      # 256x256, out_channels
        
        return out