import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ResUNet
from tuning.base_tuner import BaseTuner


def tune_resunet():

    print("\n" + "="*60)
    print("FAST TUNING: ResU-Net")
    print("="*60)

    tuner = BaseTuner(ResUNet, "ResU_Net")

    # 🔥 Hyperparameter search space
    param_space = {
        "lr": [1e-3, 1e-4],
        "batch_size": [4, 8],
        "optimizer": ["adam", "sgd"],
        "dropout": [0.1, 0.2, 0.3],
    }

    # 🔥 Run tuning
    best_params, trials = tuner.hyperparameter_search(
        param_space,
        num_trials=12,        # 🔁 you can increase to 20
        epochs_per_trial=3    # ⚡ fast evaluation
    )

    # =========================
    # 💾 SAVE TRIAL RESULTS CSV
    # =========================
    trials_df = pd.DataFrame(trials)

    save_path = os.path.join(tuner.tune_path, "resunet_tuning_results.csv")
    trials_df.to_csv(save_path, index=False)

    print(f"\n💾 Tuning results saved at: {save_path}")

    # =========================
    # 🏆 BEST PARAMS
    # =========================
    print("\n🏆 BEST PARAMETERS:")
    print(best_params)

    # =========================
    # 🔥 FINAL TRAINING
    # =========================
    results, logs = tuner.final_train(
        best_params,
        epochs=30   # full training
    )

    # =========================
    # 💾 SAVE EPOCH LOGS CSV
    # =========================
    logs_df = pd.DataFrame(logs)

    epoch_save_path = os.path.join(tuner.log_path, "resunet_epoch_logs.csv")
    logs_df.to_csv(epoch_save_path, index=False)

    print(f"\n💾 Epoch logs saved at: {epoch_save_path}")

    # =========================
    # ✅ FINAL RESULTS
    # =========================
    print("\n✅ FINAL RESULTS")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Score: {results['test_f1']:.4f}")
    print(f"Test Dice: {results['test_dice']:.4f}")

    return results


if __name__ == "__main__":
    tune_resunet()