from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_LOG_FIELDS = [
    "experiment_number",
    "model",
    "dataset_name",
    "target_column",
    "feature_columns",
    "window_size",
    "horizon",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "hidden_size",
    "learning_rate",
    "epochs",
    "mae_val",
    "rmse_val",
    "mae_test",
    "rmse_test",
    "notes",
    "run_timestamp",
]

METRICS_SUMMARY_FIELDS = [
    "run_name",
    "model",
    "split",
    "dataset_name",
    "target_column",
    "window_size",
    "horizon",
    "mae",
    "rmse",
    "run_timestamp",
]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }


def next_experiment_number(csv_path: Path) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        experiment_numbers = []
        for row in reader:
            raw_value = row.get("experiment_number", "").strip()
            if raw_value.isdigit():
                experiment_numbers.append(int(raw_value))
    return (max(experiment_numbers) + 1) if experiment_numbers else 1


def _serialize_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for fieldname in fieldnames:
        value = row.get(fieldname, "")
        if isinstance(value, (list, tuple)):
            serialized[fieldname] = " | ".join(str(item) for item in value)
        else:
            serialized[fieldname] = str(value)
    return serialized


def append_csv_row(csv_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(_serialize_row(row, fieldnames))


def build_metrics_summary_row(
    *,
    run_name: str,
    model: str,
    split: str,
    dataset_name: str,
    target_column: str,
    window_size: int,
    horizon: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "model": model,
        "split": split,
        "dataset_name": dataset_name,
        "target_column": target_column,
        "window_size": window_size,
        "horizon": horizon,
        "mae": f"{metrics['mae']:.6f}",
        "rmse": f"{metrics['rmse']:.6f}",
        "run_timestamp": _utc_timestamp(),
    }


def build_experiment_log_row(**kwargs: Any) -> dict[str, Any]:
    row = {field: "" for field in EXPERIMENT_LOG_FIELDS}
    row.update(kwargs)
    row["run_timestamp"] = row.get("run_timestamp") or _utc_timestamp()
    return row

