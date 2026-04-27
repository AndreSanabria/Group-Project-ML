from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ExperimentConfig
from src.manual_lstm import ManualLSTM, ManualLSTMConfig
from src.preprocessing import prepare_normalized_sequence_splits


@dataclass(slots=True)
class TrainingScaffold:
    train_shape: tuple[int, ...]
    val_shape: tuple[int, ...]
    test_shape: tuple[int, ...]


def run_lstm_scaffold(
    hourly_frame: pd.DataFrame, config: ExperimentConfig
) -> TrainingScaffold:
    sequence_splits = prepare_normalized_sequence_splits(hourly_frame, config)
    model_config = ManualLSTMConfig(input_size=sequence_splits.train.X.shape[2])
    ManualLSTM(model_config)

    return TrainingScaffold(
        train_shape=sequence_splits.train.X.shape,
        val_shape=sequence_splits.val.X.shape,
        test_shape=sequence_splits.test.X.shape,
    )
