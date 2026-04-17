# utils/metrics.py
import torch

class MetricsCalculator:
    """Calculate segmentation metrics"""
    
    @staticmethod
    def dice_score(pred, target, smooth=1e-6):
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * target).sum()
        return (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
    
    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)
    
    @staticmethod
    def precision(pred, target):
        pred_binary = (pred > 0.5).float()
        tp = (pred_binary * target).sum()
        fp = (pred_binary * (1 - target)).sum()
        return tp / (tp + fp + 1e-6)
    
    @staticmethod
    def recall(pred, target):
        pred_binary = (pred > 0.5).float()
        tp = (pred_binary * target).sum()
        fn = ((1 - pred_binary) * target).sum()
        return tp / (tp + fn + 1e-6)
    
    @staticmethod
    def f1_score(pred, target):
        p = MetricsCalculator.precision(pred, target)
        r = MetricsCalculator.recall(pred, target)
        return 2 * p * r / (p + r + 1e-6)
    
    @staticmethod
    def accuracy(pred, target):
        pred_binary = (pred > 0.5).float()
        correct = (pred_binary == target).sum()
        return correct / target.numel()
    
    @classmethod
    def compute_all(cls, pred, target):
        return {
            'dice': cls.dice_score(pred, target).item(),
            'iou': cls.iou_score(pred, target).item(),
            'precision': cls.precision(pred, target).item(),
            'recall': cls.recall(pred, target).item(),
            'f1': cls.f1_score(pred, target).item(),
            'accuracy': cls.accuracy(pred, target).item()
        }