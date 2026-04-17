import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MNet
from tuning.base_tuner import BaseTuner


def tune_mnet():

    print("\n" + "="*60)
    print("FAST TUNING: M-Net (GPU OPTIMIZED, 20 TRIALS)")
    print("="*60)

    tuner = BaseTuner(MNet, "M_Net")

    # 🔥 UPDATED PARAM SPACE (bigger batches for GPU usage)
    param_space = {
        "lr": [1e-3, 1e-4],
        "batch_size": [8, 16],   # 🔥 increased
        "optimizer": ["adam", "sgd"],
        "dropout": [0.1, 0.2, 0.3],
    }

    # 🚀 Hyperparameter search
    best_params, trials = tuner.hyperparameter_search(
        param_space,
        num_trials=16,
        epochs_per_trial=3
    )

    # 🔥 Save tuning results
    pd.DataFrame(trials).to_csv(
        os.path.join(tuner.tune_path, "mnet_tuning_results.csv"),
        index=False
    )

    print("\n🏆 BEST PARAMETERS:")
    print(best_params)

    # 🚀 Final training (with early stopping inside)
    results, logs = tuner.final_train(best_params, epochs=30)

    # 🔥 Save epoch logs
    pd.DataFrame(logs).to_csv(
        os.path.join(tuner.log_path, "mnet_epoch_logs.csv"),
        index=False
    )

    print("\n✅ FINAL RESULTS")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1: {results['test_f1']:.4f}")
    print(f"Test Dice: {results['test_dice']:.4f}")

    return results


if __name__ == "__main__":
    tune_mnet()