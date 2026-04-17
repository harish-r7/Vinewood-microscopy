# utils/dataset.py - FIXED VERSION
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path

class SegmentationDataset(Dataset):
    """Dataset loader for segmentation images and masks"""
    
    def __init__(self, images_dir, masks_dir):
        """
        Args:
            images_dir: Path to images folder
            masks_dir: Path to masks folder
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Get all image files
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(self.images_dir.glob(ext)))
        
        self.image_paths = sorted(self.image_paths)
        
        # Get corresponding mask paths
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / img_path.name
            if not mask_path.exists():
                # Try different extension
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = self.masks_dir / f"{img_path.stem}{ext}"
                    if test_path.exists():
                        mask_path = test_path
                        break
            self.mask_paths.append(mask_path)
        
        # Only keep valid pairs
        valid = [(img, mask) for img, mask in zip(self.image_paths, self.mask_paths) if mask.exists()]
        self.image_paths = [v[0] for v in valid]
        self.mask_paths = [v[1] for v in valid]
        
        print(f"  Loaded {len(self.image_paths)} pairs")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Load mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
        
        # Convert to binary mask (0 or 1) - FIXED: use .astype() instead of .float()
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensor
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def get_dataloaders(data_path, batch_size=8, num_workers=0, pin_memory=False):  # Set num_workers=0 to avoid multiprocessing issues
    """Create train, val, test dataloaders"""
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = SegmentationDataset(
            f"{data_path}/{split}/images",
            f"{data_path}/{split}/masks"
        )
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,  # Changed to 0 for Windows compatibility
            pin_memory=pin_memory  # Enable when using CUDA
        )
    
    return dataloaders
