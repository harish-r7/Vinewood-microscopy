# run_model.py - Complete with Train Accuracy & Epoch Logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os
import sys
from datetime import datetime

from utils.dataset import get_dataloaders
from utils.metrics import MetricsCalculator
from utils.losses import CombinedLoss
from models import *

# ========== CONFIGURATION ==========
class Config:
    DATA_PATH = "./data_prepared"
    RESULTS_PATH = "./results/before_tuning"
    CHECKPOINT_PATH = "./results/checkpoints"
    EPOCH_LOG_PATH = "./results/epoch_logs"  # NEW: Store epoch-by-epoch logs
    
    # Training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    DROPOUT = 0.3
    EARLY_STOPPING_PATIENCE = 7
    
    # Memory optimization
    GRADIENT_ACCUMULATION_STEPS = 2
    USE_MIXED_PRECISION = True
    
    # Checkpoint settings
    SAVE_CHECKPOINTS = True
    CHECKPOINT_INTERVAL = 5
    
    # Model parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    
    # GPU Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create directories
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(EPOCH_LOG_PATH, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Device: {cls.DEVICE.upper()}")
        if cls.DEVICE == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Batch Size: {cls.BATCH_SIZE} (effective: {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS})")
        print(f"Mixed Precision: {'ON' if cls.USE_MIXED_PRECISION else 'OFF'}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print("="*60)

# ========== MODEL MAPPING ==========
MODELS = {
    1: {"name": "U-Net", "class": UNet},
    2: {"name": "M-Net", "class": MNet},
    3: {"name": "Swin-Transformer", "class": SimplifiedSwin},
    4: {"name": "ResU-Net", "class": ResUNet},
    5: {"name": "DenseU-Net", "class": DenseUNet},
    6: {"name": "AttentionU-Net", "class": AttentionUNet},
    7: {"name": "MNet-MRF", "class": MNetMRF},
    8: {"name": "MNet-MRF-Voting", "class": MNetMRFVoting}
}

# ========== GPU OPTIMIZED TRAINER ==========
class GPUTrainer:
    def __init__(self, model, model_name, train_loader, val_loader, test_loader):
        self.model = model.to(Config.DEVICE)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS
        
        # Mixed precision training
        self.use_mixed_precision = Config.USE_MIXED_PRECISION and Config.DEVICE == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_mixed_precision else None
        
        self.metrics = MetricsCalculator()
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.start_epoch = 0
        self.sigmoid = nn.Sigmoid()
        
        # Store all epoch metrics
        self.epoch_metrics = []  # Will store dict for each epoch
        
        # Checkpoint paths
        model_name_clean = model_name.replace('-', '_').replace(' ', '_')
        self.checkpoint_dir = os.path.join(Config.CHECKPOINT_PATH, model_name_clean)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "last_checkpoint.pth")
        self.best_model_file = os.path.join(Config.RESULTS_PATH, f"{model_name_clean}_best.pth")
        self.epoch_log_file = os.path.join(Config.EPOCH_LOG_PATH, f"{model_name_clean}_epoch_log.csv")
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        if not Config.SAVE_CHECKPOINTS:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'model_name': self.model_name,
            'epoch_metrics': self.epoch_metrics
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_file)
        print(f"  💾 Checkpoint saved (epoch {epoch+1})")
        
        if is_best:
            torch.save(self.best_model_state, self.best_model_file)
            print(f"  ⭐ Best model saved!")
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if os.path.exists(self.checkpoint_file):
            print(f"\n📂 Found checkpoint: {self.checkpoint_file}")
            try:
                checkpoint = torch.load(self.checkpoint_file, map_location=Config.DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_val_loss = checkpoint['best_val_loss']
                self.patience_counter = checkpoint['patience_counter']
                self.epoch_metrics = checkpoint.get('epoch_metrics', [])
                
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                print(f"  ✓ Resuming from epoch {self.start_epoch + 1}")
                print(f"  ✓ Best val loss: {self.best_val_loss:.4f}")
                print(f"  ✓ Already completed {len(self.epoch_metrics)} epochs")
            except Exception as e:
                print(f"  ⚠️ Could not load checkpoint: {e}")
                self.start_epoch = 0
        else:
            print(f"\n📂 No checkpoint found. Starting fresh.")
    
    def compute_accuracy(self, outputs, masks):
        """Compute accuracy for a batch"""
        outputs_sigmoid = self.sigmoid(outputs)
        pred_binary = (outputs_sigmoid > 0.5).float()
        correct = (pred_binary == masks).sum()
        total = masks.numel()
        return (correct / total).item()
    
    def train_epoch(self):
        """Train one epoch and return metrics"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(tqdm(self.train_loader, desc="Training")):
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_accuracy += self.compute_accuracy(outputs, masks)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
        
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images = images.to(Config.DEVICE, non_blocking=True)
                masks = masks.to(Config.DEVICE, non_blocking=True)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                outputs_sigmoid = self.sigmoid(outputs)
                for i in range(images.size(0)):
                    metrics = self.metrics.compute_all(outputs_sigmoid[i], masks[i])
                    total_dice += metrics['dice']
                    total_iou += metrics['iou']
                    total_accuracy += metrics['accuracy']
                    total_precision += metrics['precision']
                    total_recall += metrics['recall']
                    total_f1 += metrics['f1']
        
        n = len(self.val_loader.dataset)
        return {
            'loss': total_loss / len(self.val_loader),
            'dice': total_dice / n,
            'iou': total_iou / n,
            'accuracy': total_accuracy / n,
            'precision': total_precision / n,
            'recall': total_recall / n,
            'f1': total_f1 / n
        }
    
    def test(self):
        """Test the model on test set"""
        self.model.eval()
        metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc="Testing"):
                images = images.to(Config.DEVICE, non_blocking=True)
                masks = masks.to(Config.DEVICE, non_blocking=True)
                
                outputs = self.model(images)
                outputs_sigmoid = self.sigmoid(outputs)
                
                for i in range(images.size(0)):
                    m = self.metrics.compute_all(outputs_sigmoid[i], masks[i])
                    for k in metrics:
                        metrics[k] += m[k]
        
        for k in metrics:
            metrics[k] /= len(self.test_loader.dataset)
        
        return metrics
    
    def save_epoch_log(self):
        """Save epoch metrics to CSV"""
        df = pd.DataFrame(self.epoch_metrics)
        df.to_csv(self.epoch_log_file, index=False)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {Config.DEVICE.upper()}")
        if Config.DEVICE == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mixed Precision: {'ON' if self.use_mixed_precision else 'OFF'}")
        print(f"Total Epochs: {Config.EPOCHS}")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(self.start_epoch, Config.EPOCHS):
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate()
                
                # Store epoch metrics
                epoch_data = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.epoch_metrics.append(epoch_data)
                
                # Save epoch log after each epoch
                self.save_epoch_log()
                
                # Print progress
                print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS}")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
                print(f"  Val Dice:   {val_metrics['dice']:.4f} | Val IoU:   {val_metrics['iou']:.4f}")
                
                # GPU memory info
                if Config.DEVICE == 'cuda':
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"  GPU Memory: {mem_used:.2f} GB")
                
                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    print(f"  🎉 New best model! (Loss: {val_metrics['loss']:.4f})")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement for {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE} epochs")
                
                # Save checkpoint
                if (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(epoch, is_best)
                
                # Early stopping
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    break
                
                print("-" * 60)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Keyboard interrupt detected!")
            print(f"Saving checkpoint before exit...")
            self.save_checkpoint(epoch)
            self.save_epoch_log()
            print(f"✓ Checkpoint and epoch log saved. Run again to resume training.")
            sys.exit(0)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final best model
        torch.save(self.best_model_state, self.best_model_file)
        print(f"\n✓ Best model saved: {self.best_model_file}")
        
        # Test
        print(f"\n{'='*60}")
        print(f"TESTING {self.model_name}")
        print(f"{'='*60}")
        test_metrics = self.test()
        
        # Save final epoch log
        self.save_epoch_log()
        print(f"✓ Epoch log saved to: {self.epoch_log_file}")
        
        return test_metrics

# ========== ANALYSIS FUNCTIONS ==========
def analyze_results(model_name):
    """Analyze the training results"""
    model_name_clean = model_name.replace('-', '_').replace(' ', '_')
    epoch_log_file = os.path.join(Config.EPOCH_LOG_PATH, f"{model_name_clean}_epoch_log.csv")
    
    if not os.path.exists(epoch_log_file):
        print(f"No epoch log found for {model_name}")
        return
    
    df = pd.read_csv(epoch_log_file)
    
    print("\n" + "="*60)
    print(f"ANALYSIS FOR {model_name}")
    print("="*60)
    
    # Best metrics
    best_val_dice = df['val_dice'].max()
    best_val_acc = df['val_accuracy'].max()
    best_epoch_dice = df['val_dice'].idxmax() + 1
    best_epoch_acc = df['val_accuracy'].idxmax() + 1
    
    print(f"\n📈 BEST METRICS:")
    print(f"  Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch_dice})")
    print(f"  Best Val Acc:  {best_val_acc:.4f} (Epoch {best_epoch_acc})")
    
    # Training progression
    print(f"\n📊 TRAINING PROGRESSION:")
    print(f"  Start - Train Loss: {df['train_loss'].iloc[0]:.4f} → End: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Start - Val Loss:   {df['val_loss'].iloc[0]:.4f} → End: {df['val_loss'].iloc[-1]:.4f}")
    print(f"  Start - Val Dice:   {df['val_dice'].iloc[0]:.4f} → End: {df['val_dice'].iloc[-1]:.4f}")
    
    # Overfitting check
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    loss_gap = final_val_loss - final_train_loss
    
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    if loss_gap > 0.1:
        print(f"  ⚠️ Possible overfitting! (Val loss > Train loss by {loss_gap:.4f})")
    elif loss_gap < -0.1:
        print(f"  ✓ Underfitting (Val loss < Train loss by {abs(loss_gap):.4f})")
    else:
        print(f"  ✓ Good balance! (Loss gap: {loss_gap:.4f})")
    
    # Early stopping
    if len(df) < Config.EPOCHS:
        print(f"\n⏹️ Early stopping triggered at epoch {len(df)}")
    
    return df

# ========== MAIN ==========
def main():
    Config.print_config()
    
    # Select model
    MODEL_NUMBER = 8  # 🔧 CHANGE THIS (1-8)
    
    if MODEL_NUMBER not in MODELS:
        print(f"❌ Invalid model number! Use 1-8")
        return
    
    model_info = MODELS[MODEL_NUMBER]
    model_name = model_info["name"]
    model_class = model_info["class"]
    
    print(f"\n✅ Selected: {model_name}")
    
    # Load data
    print("\n📂 Loading datasets...")
    dataloaders = get_dataloaders(Config.DATA_PATH, Config.BATCH_SIZE, num_workers=0)
    
    print(f"  Train: {len(dataloaders['train'].dataset)} images")
    print(f"  Val:   {len(dataloaders['val'].dataset)} images")
    print(f"  Test:  {len(dataloaders['test'].dataset)} images")
    
    # Create model
    print(f"\n🔧 Creating model...")
    model = model_class(Config.IN_CHANNELS, Config.OUT_CHANNELS, Config.DROPOUT)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train
    trainer = GPUTrainer(
        model, model_name,
        dataloaders['train'],
        dataloaders['val'],
        dataloaders['test']
    )
    
    test_metrics = trainer.train()
    
    # Save final results
    print(f"\n{'='*60}")
    print(f"✅ FINAL RESULTS FOR {model_name}")
    print(f"{'='*60}")
    for key, value in test_metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    # Save to CSV
    results_df = pd.DataFrame([{
        'model': model_name,
        'device': Config.DEVICE,
        'batch_size': Config.BATCH_SIZE,
        'effective_batch_size': Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS,
        'mixed_precision': Config.USE_MIXED_PRECISION,
        'epochs_trained': len(trainer.epoch_metrics),
        'best_val_loss': trainer.best_val_loss,
        'test_accuracy': test_metrics['accuracy'],
        'test_dice': test_metrics['dice'],
        'test_iou': test_metrics['iou'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    csv_path = os.path.join(Config.RESULTS_PATH, f"{model_name.replace('-', '_').replace(' ', '_')}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Analyze results
    analyze_results(model_name)
    
    # Clear GPU cache
    if Config.DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f"\n✓ GPU cache cleared")
    
    return test_metrics

if __name__ == "__main__":
    results = main()