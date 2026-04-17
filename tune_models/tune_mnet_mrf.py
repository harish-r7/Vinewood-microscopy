# tune_models/tune_mnet_mrf.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MNetMRF
from tuning.base_tuner import BaseTuner


def tune_mnet_mrf():

    print("\n" + "="*60)
    print("FAST TUNING: MNet-MRF (GPU + EARLY STOPPING)")
    print("="*60)

    tuner = BaseTuner(MNetMRF, "MNet_MRF")

    # 🔥 Parameter space (optimized)
    param_space = {
        "lr": [1e-2, 1e-3, 1e-4],     # 👈 includes 0.01 as you asked
        "batch_size": [8, 16],        # 👈 safe for RAM
        "optimizer": ["adam", "sgd"],
        "dropout": [0.1, 0.3],
    }

    # 🔥 Hyperparameter tuning
    best_params, trials = tuner.hyperparameter_search(
        param_space,
        num_trials=20,
        epochs_per_trial=3
    )

    print("\n🏆 BEST PARAMETERS FOUND:")
    print(best_params)

    # 🔥 Final training with early stopping + full logs
    results, logs = tuner.final_train(
        best_params,
        epochs=30
    )

    print("\n✅ FINAL RESULTS")
    print(results)

    return results


if __name__ == "__main__":
    tune_mnet_mrf()