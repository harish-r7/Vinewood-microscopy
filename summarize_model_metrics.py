from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

BEFORE_MODEL_NAMES = [
    "AttentionU_Net",
    "DenseU_Net",
    "MNet_MRF",
    "MNet_MRF_Voting",
    "M_Net",
    "ResU_Net",
    "Swin_Transformer",
    "U_Net",
]

AFTER_MODEL_FILES = {
    "DenseU_Net": RESULTS_DIR / "after_tuning" / "DenseU_Net" / "epoch_logs" / "denseunet_epoch_logs.csv",
    "MNet_MRF": RESULTS_DIR / "after_tuning" / "MNet_MRF" / "epoch_logs" / "epoch_logs.csv",
    "M_Net": RESULTS_DIR / "after_tuning" / "M_Net" / "epoch_logs" / "mnet_epoch_logs.csv",
    "ResU_Net": RESULTS_DIR / "after_tuning" / "ResU_Net" / "epoch_logs" / "resunet_epoch_logs.csv",
    "Swin_Transformer": RESULTS_DIR / "after_tuning" / "Swin_Transformer_FINAL" / "epoch_logs" / "epoch_log.csv",
    "U_Net": RESULTS_DIR / "after_tuning" / "U_Net" / "epoch_logs" / "training_log.csv",
}

OUTPUT_FILE = RESULTS_DIR / "all_models_metrics_ranked.csv"
BEFORE_OUTPUT_FILE = RESULTS_DIR / "before_tuning_models_metrics_ranked.csv"
AFTER_OUTPUT_FILE = RESULTS_DIR / "after_tuning_models_metrics_ranked.csv"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_model_name(name: str) -> str:
    mapping = {
        "U_Net": "U-Net",
        "ResU_Net": "ResU-Net",
        "DenseU_Net": "DenseU-Net",
        "AttentionU_Net": "AttentionU-Net",
        "M_Net": "M-Net",
        "MNet_MRF": "MNet-MRF",
        "MNet_MRF_Voting": "MNet-MRF-Voting",
        "Swin_Transformer": "Swin-Transformer",
    }
    return mapping.get(name, name.replace("_", "-"))


def fitting_status(final_train_loss: float | None, final_val_loss: float | None) -> tuple[str, float | None]:
    if final_train_loss is None or final_val_loss is None:
        return "Unknown", None
    loss_gap = final_val_loss - final_train_loss
    if loss_gap > 0.1:
        return "OVERFITTING", loss_gap
    if loss_gap < -0.1:
        return "UNDERFITTING", loss_gap
    return "Balanced", loss_gap


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            as_float(row.get("val_dice")) or as_float(row.get("dice")) or float("-inf"),
            as_float(row.get("val_accuracy")) or float("-inf"),
        ),
    )


def build_before_row(model_key: str) -> dict[str, Any]:
    epoch_log = RESULTS_DIR / "epoch_logs" / f"{model_key}_epoch_log.csv"
    results_csv = RESULTS_DIR / "before_tuning" / f"{model_key}_results.csv"
    rows = read_csv_rows(epoch_log)
    last = rows[-1]
    best = best_row(rows)
    test_row = read_csv_rows(results_csv)[0]
    status, loss_gap = fitting_status(as_float(last.get("train_loss")), as_float(last.get("val_loss")))

    return {
        "model_name": normalize_model_name(model_key),
        "model_key": model_key,
        "phase": "Before Tuning",
        "is_tuned": "No",
        "epochs_trained": int(float(test_row["epochs_trained"])),
        "selected_epoch": int(float(best["epoch"])),
        "final_train_loss": as_float(last.get("train_loss")),
        "final_train_accuracy": as_float(last.get("train_accuracy")),
        "selected_train_accuracy": as_float(best.get("train_accuracy")),
        "final_val_loss": as_float(last.get("val_loss")),
        "best_val_loss": as_float(test_row.get("best_val_loss")) or min(as_float(row.get("val_loss")) or float("inf") for row in rows),
        "final_val_accuracy": as_float(last.get("val_accuracy")),
        "best_val_accuracy": max(as_float(row.get("val_accuracy")) or float("-inf") for row in rows),
        "final_val_precision": as_float(last.get("val_precision")),
        "final_val_recall": as_float(last.get("val_recall")),
        "final_val_f1": as_float(last.get("val_f1_score")),
        "final_val_dice": as_float(last.get("val_dice")),
        "best_val_dice": max(as_float(row.get("val_dice")) or float("-inf") for row in rows),
        "test_loss": None,
        "test_accuracy": as_float(test_row.get("test_accuracy")),
        "test_precision": as_float(test_row.get("test_precision")),
        "test_recall": as_float(test_row.get("test_recall")),
        "test_f1": as_float(test_row.get("test_f1")),
        "test_dice": as_float(test_row.get("test_dice")),
        "test_iou": as_float(test_row.get("test_iou")),
        "loss_gap": loss_gap,
        "fitting_status": status,
        "sort_score": as_float(test_row.get("test_accuracy")) or -1.0,
    }


def build_after_row(model_key: str, path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    last = rows[-1]
    best = best_row(rows)
    status, loss_gap = fitting_status(as_float(last.get("train_loss")), as_float(last.get("val_loss")))

    return {
        "model_name": normalize_model_name(model_key),
        "model_key": model_key,
        "phase": "After Tuning",
        "is_tuned": "Yes",
        "epochs_trained": len(rows),
        "selected_epoch": int(float(best["epoch"])),
        "final_train_loss": as_float(last.get("train_loss")),
        "final_train_accuracy": as_float(last.get("train_accuracy")),
        "selected_train_accuracy": as_float(best.get("train_accuracy")),
        "final_val_loss": as_float(last.get("val_loss")),
        "best_val_loss": min(as_float(row.get("val_loss")) or float("inf") for row in rows),
        "final_val_accuracy": as_float(last.get("val_accuracy")),
        "best_val_accuracy": max(as_float(row.get("val_accuracy")) or float("-inf") for row in rows),
        "final_val_precision": as_float(last.get("val_precision")) or as_float(last.get("precision")),
        "final_val_recall": as_float(last.get("val_recall")) or as_float(last.get("recall")),
        "final_val_f1": as_float(last.get("val_f1_score")) or as_float(last.get("val_f1")) or as_float(last.get("f1_score")),
        "final_val_dice": as_float(last.get("val_dice")) or as_float(last.get("dice")),
        "best_val_dice": max(
            as_float(row.get("val_dice")) or as_float(row.get("dice")) or float("-inf")
            for row in rows
        ),
        "test_loss": as_float(best.get("test_loss")),
        "test_accuracy": as_float(best.get("test_accuracy")),
        "test_precision": as_float(best.get("test_precision")),
        "test_recall": as_float(best.get("test_recall")),
        "test_f1": as_float(best.get("test_f1_score")) or as_float(best.get("test_f1")),
        "test_dice": as_float(best.get("test_dice")),
        "test_iou": None,
        "loss_gap": loss_gap,
        "fitting_status": status,
        "sort_score": as_float(best.get("test_accuracy")) or -1.0,
    }


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            return ""
        return f"{value:.6f}"
    return value


def ranked_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(record) for record in records]
    ranked.sort(
        key=lambda row: (
            row["sort_score"],
            row["test_dice"] if row["test_dice"] is not None else -1.0,
            row["best_val_accuracy"] if row["best_val_accuracy"] is not None else -1.0,
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["overall_rank"] = rank
        row.pop("sort_score", None)
    return ranked


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})


def try_write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> tuple[bool, str | None]:
    try:
        write_csv(path, records, fieldnames)
        return True, None
    except PermissionError as exc:
        return False, str(exc)


def main() -> None:
    before_records = [build_before_row(model_key) for model_key in BEFORE_MODEL_NAMES]
    after_records = [build_after_row(model_key, path) for model_key, path in AFTER_MODEL_FILES.items()]
    records = before_records + after_records

    ranked_all = ranked_records(records)
    ranked_before = ranked_records(before_records)
    ranked_after = ranked_records(after_records)

    fieldnames = [
        "overall_rank",
        "model_name",
        "model_key",
        "phase",
        "is_tuned",
        "epochs_trained",
        "selected_epoch",
        "final_train_loss",
        "final_train_accuracy",
        "selected_train_accuracy",
        "final_val_loss",
        "best_val_loss",
        "final_val_accuracy",
        "best_val_accuracy",
        "final_val_precision",
        "final_val_recall",
        "final_val_f1",
        "final_val_dice",
        "best_val_dice",
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_dice",
        "test_iou",
        "loss_gap",
        "fitting_status",
    ]

    output_statuses = [
        ("ranked summary", OUTPUT_FILE, *try_write_csv(OUTPUT_FILE, ranked_all, fieldnames)),
        ("before-tuning summary", BEFORE_OUTPUT_FILE, *try_write_csv(BEFORE_OUTPUT_FILE, ranked_before, fieldnames)),
        ("after-tuning summary", AFTER_OUTPUT_FILE, *try_write_csv(AFTER_OUTPUT_FILE, ranked_after, fieldnames)),
    ]

    for label, path, ok, error in output_statuses:
        if ok:
            print(f"Saved {label} to: {path}")
        else:
            print(f"Skipped {label}: {error}")


if __name__ == "__main__":
    main()
