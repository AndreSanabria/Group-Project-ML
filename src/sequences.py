from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray


def create_sliding_window_dataset(
    frame: pd.DataFrame,
    target_column: str,
    window_size: int,
    horizon: int = 1,
) -> SequenceDataset:
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")
    if window_size <= 0 or horizon <= 0:
        raise ValueError("window_size and horizon must both be positive integers.")

    total_rows = len(frame)
    sample_count = total_rows - window_size - horizon + 1
    if sample_count <= 0:
        raise ValueError(
            "Not enough rows to build sequences. "
            f"Need at least {window_size + horizon} rows, found {total_rows}."
        )

    values = frame.to_numpy(dtype=float)
    target_index = frame.columns.get_loc(target_column)
    feature_count = values.shape[1]

    X = np.zeros((sample_count, window_size, feature_count), dtype=float)
    y = np.zeros(sample_count, dtype=float)
    timestamps = np.zeros(sample_count, dtype="datetime64[ns]")

    for sample_idx in range(sample_count):
        window_start = sample_idx
        window_end = sample_idx + window_size
        target_position = window_end + horizon - 1

        X[sample_idx] = values[window_start:window_end]
        y[sample_idx] = values[target_position, target_index]
        timestamps[sample_idx] = frame.index[target_position]

    return SequenceDataset(X=X, y=y, timestamps=timestamps)

