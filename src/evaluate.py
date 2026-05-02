from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import REPO_ROOT

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
    "mae_train",
    "rmse_train",
    "mae_val",
    "rmse_val",
    "mae_test",
    "rmse_test",
    "plot_path",
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
    true_values, predicted_values = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean(np.abs(true_values - predicted_values)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values, predicted_values = _validate_regression_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((true_values - predicted_values) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values, predicted_values = _validate_regression_inputs(y_true, y_pred)
    return {
        "mae": mean_absolute_error(true_values, predicted_values),
        "rmse": root_mean_squared_error(true_values, predicted_values),
    }


def _validate_regression_inputs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape after flattening. "
            f"Got {true_values.shape} and {predicted_values.shape}."
        )
    if true_values.size == 0:
        raise ValueError("Cannot compute regression metrics for an empty array.")
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Regression metric inputs must contain only finite values.")

    return true_values, predicted_values


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
    serialized_row = _serialize_row(row, fieldnames)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_fieldnames != fieldnames:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(_serialize_row(existing_row, fieldnames))

    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(serialized_row)


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


def format_path_for_log(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path_obj)
