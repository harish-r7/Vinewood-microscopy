import os
import random
import torch
import pandas as pd

from utils.dataset import get_dataloaders
from .config import Config


# ---------- METRICS ----------
def calculate_metrics(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum().float()
    tn = ((1 - pred) * (1 - target)).sum().float()
    fp = (pred * (1 - target)).sum().float()
    fn = ((1 - pred) * target).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    return accuracy.item(), precision.item(), recall.item(), f1.item(), dice.item()


class BaseTuner:

    def __init__(self, model_class, model_name):
        self.model_class = model_class
        self.model_name = model_name

        # 🔥 FORCE GPU (since you confirmed CUDA works)
        self.device = torch.device("cuda")

        print(f"\n🚀 Using Device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

        self.base_path = os.path.join(Config.AFTER_TUNING_PATH, model_name)
        self.log_path = os.path.join(self.base_path, "epoch_logs")
        self.tune_path = os.path.join(self.base_path, "tuning_logs")

        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.tune_path, exist_ok=True)

    # =========================
    # 🚀 HYPERPARAMETER SEARCH
    # =========================
    def hyperparameter_search(self, param_space, num_trials=20, epochs_per_trial=3):

        print("\n" + "="*60)
        print("FAST TUNING")
        print("="*60)

        trials = []

        for i in range(num_trials):

            params = {k: random.choice(v) for k, v in param_space.items()}

            print(f"\n🚀 Trial {i+1}/{num_trials}")
            print(params)

            val_loss = self._single_run(params, epochs_per_trial)

            print(f"Validation Loss: {val_loss:.4f}")

            trials.append({**params, "val_loss": val_loss})

        best = min(trials, key=lambda x: x["val_loss"])

        return best, trials

    # =========================
    # 🔥 FAST RUN
    # =========================
    def _single_run(self, params, epochs):

        loaders = get_dataloaders(
            Config.DATA_PATH,
            params["batch_size"],
            num_workers=4,
            pin_memory=True
        )

        model = self.model_class(
            Config.IN_CHANNELS,
            Config.OUT_CHANNELS,
            dropout=params["dropout"]
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(epochs):

            print(f"   Epoch {epoch+1}/{epochs}")
            model.train()

            for x, y in loaders["train"]:

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)

                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y in loaders["val"]:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item()

            val_loss /= len(loaders["val"])
            print(f"   Val Loss: {val_loss:.4f}")

        return val_loss

    # =========================
    # 🔥 FINAL TRAINING
    # =========================
    def final_train(self, best_params, epochs=30):

        loaders = get_dataloaders(
            Config.DATA_PATH,
            best_params["batch_size"],
            num_workers=2,
            pin_memory=True
        )

        model = self.model_class(
            Config.IN_CHANNELS,
            Config.OUT_CHANNELS,
            dropout=best_params["dropout"]
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience = 5
        counter = 0
        min_delta = 0.01

        logs = []

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            # DEBUG (only first epoch)
            if epoch == 0:
                print("DEBUG DEVICE CHECK:")
                for x, y in loaders["train"]:
                    print("Input device:", x.device)
                    print("Model device:", next(model.parameters()).device)
                    break

            # TRAIN
            model.train()
            train_loss = train_acc = 0

            for x, y in loaders["train"]:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)

                pred = (torch.sigmoid(out) > 0.5).float()
                acc, _, _, _, _ = calculate_metrics(pred, y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc

            train_loss /= len(loaders["train"])
            train_acc /= len(loaders["train"])

            # VALIDATION
            model.eval()
            val_loss = val_acc = val_prec = val_rec = val_f1 = val_dice = 0

            with torch.no_grad():
                for x, y in loaders["val"]:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    out = model(x)
                    loss = criterion(out, y)

                    pred = (torch.sigmoid(out) > 0.5).float()
                    acc, prec, rec, f1, dice = calculate_metrics(pred, y)

                    val_loss += loss.item()
                    val_acc += acc
                    val_prec += prec
                    val_rec += rec
                    val_f1 += f1
                    val_dice += dice

            val_loss /= len(loaders["val"])
            val_acc /= len(loaders["val"])
            val_prec /= len(loaders["val"])
            val_rec /= len(loaders["val"])
            val_f1 /= len(loaders["val"])
            val_dice /= len(loaders["val"])

            # TEST
            test_loss = test_acc = test_prec = test_rec = test_f1 = test_dice = 0

            with torch.no_grad():
                for x, y in loaders["test"]:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    out = model(x)
                    loss = criterion(out, y)

                    pred = (torch.sigmoid(out) > 0.5).float()
                    acc, prec, rec, f1, dice = calculate_metrics(pred, y)

                    test_loss += loss.item()
                    test_acc += acc
                    test_prec += prec
                    test_rec += rec
                    test_f1 += f1
                    test_dice += dice

            test_loss /= len(loaders["test"])
            test_acc /= len(loaders["test"])
            test_prec /= len(loaders["test"])
            test_rec /= len(loaders["test"])
            test_f1 /= len(loaders["test"])
            test_dice /= len(loaders["test"])

            print(f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")

            logs.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1_score": val_f1,
                "val_dice": val_dice,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1_score": test_f1,
                "test_dice": test_dice
            })

            # EARLY STOPPING
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            print(f"EarlyStopping: {counter}/{patience}")

            if counter >= patience:
                print("⛔ Early stopping triggered")
                break

        # SAVE CSV
        pd.DataFrame(logs).to_csv(
            os.path.join(self.log_path, "epoch_logs.csv"),
            index=False
        )

        return {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_dice": test_dice
        }, logs