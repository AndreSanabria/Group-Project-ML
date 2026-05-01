from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _limit_series(
    y_true: np.ndarray, y_pred: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape after flattening. "
            f"Got {true_values.shape} and {predicted_values.shape}."
        )
    if max_points <= 0:
        raise ValueError("max_points must be a positive integer.")

    sample_count = min(len(true_values), max_points)
    x_axis = np.arange(sample_count)
    return x_axis, true_values[:sample_count], predicted_values[:sample_count]


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    max_points: int = 200,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_axis, true_values, predicted_values = _limit_series(y_true, y_pred, max_points)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(x_axis, true_values, label="Actual", linewidth=2)
    axis.plot(x_axis, predicted_values, label="Predicted", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Temperature")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    max_points: int = 200,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_axis, true_values, predicted_values = _limit_series(y_true, y_pred, max_points)
    residuals = true_values - predicted_values

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axis.plot(x_axis, residuals, label="Actual - Predicted", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Temperature error")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

