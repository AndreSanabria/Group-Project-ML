from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    max_points: int = 200,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = min(len(y_true), max_points)
    x_axis = np.arange(sample_count)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(x_axis, y_true[:sample_count], label="Actual", linewidth=2)
    axis.plot(x_axis, y_pred[:sample_count], label="Predicted", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Temperature")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

