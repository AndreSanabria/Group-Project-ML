from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_SENTINEL_REPLACEMENTS, ExperimentConfig


@dataclass(slots=True)
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(slots=True)
class NormalizationStats:
    means: pd.Series
    stds: pd.Series


@dataclass(slots=True)
class PreparedSequenceSplits:
    train: "SequenceDataset"
    val: "SequenceDataset"
    test: "SequenceDataset"
    normalization_stats: NormalizationStats


def replace_sentinel_values(
    frame: pd.DataFrame,
    sentinel_replacements: dict[str, tuple[float, ...]] = DEFAULT_SENTINEL_REPLACEMENTS,
    replacement_value: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned_frame = frame.copy()
    replacement_counts: dict[str, int] = {}

    for column, sentinel_values in sentinel_replacements.items():
        if column not in cleaned_frame.columns:
            continue
        column_mask = cleaned_frame[column].isin(sentinel_values)
        replacement_counts[column] = int(column_mask.sum())
        if replacement_counts[column]:
            cleaned_frame.loc[column_mask, column] = replacement_value

    return cleaned_frame, replacement_counts


def resample_to_hourly(
    frame: pd.DataFrame, hours: int = 1, aggregation: str = "mean"
) -> pd.DataFrame:
    if hours <= 0:
        raise ValueError("hours must be a positive integer.")

    resample_rule = f"{hours}h"
    resampler = frame.resample(resample_rule)

    if aggregation == "mean":
        resampled_frame = resampler.mean(numeric_only=True)
    elif aggregation == "median":
        resampled_frame = resampler.median(numeric_only=True)
    else:
        raise ValueError("aggregation must be either 'mean' or 'median'.")

    return resampled_frame.dropna(how="all")


def select_feature_columns(
    frame: pd.DataFrame, feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the dataset: "
            f"{missing}. Available columns: {list(frame.columns)}"
        )
    return frame.loc[:, list(feature_columns)].dropna()


def summarize_time_series_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "start_timestamp": None,
            "end_timestamp": None,
            "missing_values": {column: 0 for column in frame.columns},
            "statistics": {},
        }

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "start_timestamp": frame.index.min().isoformat(),
        "end_timestamp": frame.index.max().isoformat(),
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().to_dict().items()
        },
        "statistics": {
            column: {
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std(ddof=0)),
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
            for column in frame.columns
        },
    }


def save_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def preprocess_hourly_dataset(
    raw_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    resample_hours: int = 1,
    aggregation: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned_frame, sentinel_counts = replace_sentinel_values(raw_frame)
    hourly_frame = resample_to_hourly(
        cleaned_frame, hours=resample_hours, aggregation=aggregation
    )
    processed_frame = select_feature_columns(hourly_frame, feature_columns)

    summary = {
        "resample_hours": resample_hours,
        "aggregation": aggregation,
        "feature_columns": list(feature_columns),
        "sentinel_replacements": sentinel_counts,
        "raw_summary": summarize_time_series_frame(raw_frame),
        "cleaned_summary": summarize_time_series_frame(cleaned_frame),
        "processed_summary": summarize_time_series_frame(processed_frame),
    }
    return processed_frame, summary


def chronological_split(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> DataSplits:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Train/validation/test ratios must sum to 1.0.")

    total_rows = len(frame)
    if total_rows < 3:
        raise ValueError("The dataset is too small to split chronologically.")

    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)

    train = frame.iloc[:train_end].copy()
    val = frame.iloc[train_end:val_end].copy()
    test = frame.iloc[val_end:].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "One of the chronological splits is empty. "
            "Adjust the split ratios or use more data."
        )

    return DataSplits(train=train, val=val, test=test)


def fit_standardizer(train_frame: pd.DataFrame) -> NormalizationStats:
    means = train_frame.mean()
    stds = train_frame.std(ddof=0).replace(0, 1.0)
    return NormalizationStats(means=means, stds=stds)


def apply_standardizer(
    frame: pd.DataFrame, normalization_stats: NormalizationStats
) -> pd.DataFrame:
    return (frame - normalization_stats.means) / normalization_stats.stds


def prepare_normalized_sequence_splits(
    hourly_frame: pd.DataFrame, config: ExperimentConfig
) -> PreparedSequenceSplits:
    from src.sequences import create_sliding_window_dataset
    from src.sequences import subset_sequence_dataset

    selected_frame = select_feature_columns(hourly_frame, config.feature_columns)
    raw_splits = chronological_split(
        selected_frame,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    normalization_stats = fit_standardizer(raw_splits.train)
    normalized_frame = apply_standardizer(selected_frame, normalization_stats)
    full_sequence_dataset = create_sliding_window_dataset(
        normalized_frame, config.target_column, config.window_size, config.horizon
    )

    train_end_timestamp = np.datetime64(raw_splits.train.index[-1].to_datetime64())
    val_end_timestamp = np.datetime64(raw_splits.val.index[-1].to_datetime64())

    train_mask = full_sequence_dataset.timestamps <= train_end_timestamp
    val_mask = (
        (full_sequence_dataset.timestamps > train_end_timestamp)
        & (full_sequence_dataset.timestamps <= val_end_timestamp)
    )
    test_mask = full_sequence_dataset.timestamps > val_end_timestamp

    return PreparedSequenceSplits(
        train=subset_sequence_dataset(full_sequence_dataset, train_mask),
        val=subset_sequence_dataset(full_sequence_dataset, val_mask),
        test=subset_sequence_dataset(full_sequence_dataset, test_mask),
        normalization_stats=normalization_stats,
    )
