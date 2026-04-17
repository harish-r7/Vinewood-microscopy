import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SimplifiedSwin
from tuning.base_tuner import BaseTuner

torch.backends.cudnn.benchmark = True


def run_final_training():
    print("\n" + "="*60)
    print("FINAL TRAINING (FULL METRICS + TEST + RESUME)")
    print("="*60)

    best_params = {
        'lr': 0.001,
        'batch_size': 8,
        'loss': 'bce_dice',
        'optimizer': 'adam',
        'dropout': 0.1
    }

    tuner = BaseTuner(SimplifiedSwin, "Swin_Transformer_FINAL")

    results = tuner.final_train(best_params)

    print("\n✅ FINAL RESULTS")
    print(results)


if __name__ == "__main__":
    run_final_training()