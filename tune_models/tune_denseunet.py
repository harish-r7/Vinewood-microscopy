import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DenseUNet
from tuning.base_tuner import BaseTuner


def tune_denseunet():

    print("\n" + "="*60)
    print("FAST TUNING: DenseU-Net (16 TRIALS)")
    print("="*60)

    tuner = BaseTuner(DenseUNet, "DenseU_Net")

    # 🔥 Hyperparameter search space
    param_space = {
        "lr": [1e-3, 1e-4],
        "batch_size": [4, 8],
        "optimizer": ["adam", "sgd"],
        "dropout": [0.1, 0.2, 0.3],
    }

    # =========================
    # 🚀 HYPERPARAMETER SEARCH
    # =========================
    best_params, trials = tuner.hyperparameter_search(
        param_space,
        num_trials=16,        # 🔥 CHANGED TO 16
        epochs_per_trial=3    # ⚡ fast tuning
    )

    # =========================
    # 💾 SAVE TUNING CSV
    # =========================
    trials_df = pd.DataFrame(trials)

    tuning_path = os.path.join(tuner.tune_path, "denseunet_tuning_results.csv")
    trials_df.to_csv(tuning_path, index=False)

    print(f"\n💾 Tuning results saved at: {tuning_path}")

    # =========================
    # 🏆 BEST PARAMETERS
    # =========================
    print("\n🏆 BEST PARAMETERS:")
    print(best_params)

    # =========================
    # 🔥 FINAL TRAINING
    # =========================
    results, logs = tuner.final_train(
        best_params,
        epochs=30
    )

    # =========================
    # 💾 SAVE EPOCH CSV
    # =========================
    logs_df = pd.DataFrame(logs)

    epoch_path = os.path.join(tuner.log_path, "denseunet_epoch_logs.csv")
    logs_df.to_csv(epoch_path, index=False)

    print(f"\n💾 Epoch logs saved at: {epoch_path}")

    # =========================
    # ✅ FINAL RESULTS
    # =========================
    print("\n✅ FINAL RESULTS")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Score: {results['test_f1']:.4f}")
    print(f"Test Dice: {results['test_dice']:.4f}")

    return results


if __name__ == "__main__":
    tune_denseunet()