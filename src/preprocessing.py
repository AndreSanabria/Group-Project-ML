from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ExperimentConfig


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


def resample_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    hourly_frame = frame.resample("1h").mean(numeric_only=True)
    return hourly_frame.dropna(how="all")


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

    selected_frame = select_feature_columns(hourly_frame, config.feature_columns)
    raw_splits = chronological_split(
        selected_frame,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    normalization_stats = fit_standardizer(raw_splits.train)

    normalized_train = apply_standardizer(raw_splits.train, normalization_stats)
    normalized_val = apply_standardizer(raw_splits.val, normalization_stats)
    normalized_test = apply_standardizer(raw_splits.test, normalization_stats)

    return PreparedSequenceSplits(
        train=create_sliding_window_dataset(
            normalized_train, config.target_column, config.window_size, config.horizon
        ),
        val=create_sliding_window_dataset(
            normalized_val, config.target_column, config.window_size, config.horizon
        ),
        test=create_sliding_window_dataset(
            normalized_test, config.target_column, config.window_size, config.horizon
        ),
        normalization_stats=normalization_stats,
    )

