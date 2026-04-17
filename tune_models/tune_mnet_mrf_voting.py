# tune_models/tune_mnet_mrf_voting.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MNetMRFVoting
from tuning.base_tuner import BaseTuner
from tuning.config import Config
from utils.dataset import get_dataloaders

def tune_mnet_mrf_voting():
    print("\n" + "="*60)
    print("TUNING: MNet-MRF-Voting")
    print("="*60)
    
    tuner = BaseTuner(MNetMRFVoting, "MNet_MRF_Voting")
    base_loaders = get_dataloaders(Config.DATA_PATH, Config.DEFAULT_BATCH_SIZE, num_workers=0)
    
    best_lr = tuner.tune_lr(base_loaders['train'], base_loaders['val'])
    best_batch = tuner.tune_batch(best_lr, base_loaders['val'], base_loaders['test'])
    best_loss = tuner.tune_loss(best_lr, best_batch, base_loaders['val'], base_loaders['test'])
    best_dropout = tuner.tune_dropout(best_lr, best_batch, best_loss, base_loaders['val'], base_loaders['test'])
    best_optimizer = tuner.tune_optimizer(best_lr, best_batch, best_loss, best_dropout, base_loaders['val'], base_loaders['test'])
    
    best_params = {'lr': best_lr, 'batch_size': best_batch, 'loss': best_loss, 'optimizer': best_optimizer, 'dropout': best_dropout}
    print(f"\n📊 Best Parameters: {best_params}")
    
    results = tuner.final_train(best_params)
    print(f"\n✅ Tuning complete! Test Acc: {results['test_accuracy']:.4f}, Test Dice: {results['test_dice']:.4f}")
    return results

if __name__ == "__main__":
    tune_mnet_mrf_voting()