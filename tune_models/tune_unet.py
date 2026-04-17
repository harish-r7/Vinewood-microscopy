import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet
from tuning.base_tuner import BaseTuner
from tuning.config import Config


def tune_unet():

    print("\n" + "="*60)
    print("FAST TUNING: U-NET")
    print("="*60)

    tuner = BaseTuner(UNet, "UNet")

    param_space = {
        "lr": [1e-3, 1e-4],
        "batch_size": [4, 8],
        "optimizer": ["adam", "sgd"],
        "dropout": [0.1, 0.3],
    }

    best_params, trials = tuner.hyperparameter_search(
        param_space,
        num_trials=12,        # 👈 CHANGE HERE
        epochs_per_trial=3    # 👈 CHANGE HERE
    )

    print("\n🏆 BEST PARAMS:")
    print(best_params)

    results, logs = tuner.final_train(best_params, epochs=30)

    print("\n✅ FINAL RESULTS")
    print(results)


if __name__ == "__main__":
    tune_unet()