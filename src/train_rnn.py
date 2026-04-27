from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ExperimentConfig
from src.manual_rnn import ManualRNN, ManualRNNConfig
from src.preprocessing import prepare_normalized_sequence_splits


@dataclass(slots=True)
class TrainingScaffold:
    train_shape: tuple[int, ...]
    val_shape: tuple[int, ...]
    test_shape: tuple[int, ...]


def run_rnn_scaffold(hourly_frame: pd.DataFrame, config: ExperimentConfig) -> TrainingScaffold:
    sequence_splits = prepare_normalized_sequence_splits(hourly_frame, config)
    model_config = ManualRNNConfig(input_size=sequence_splits.train.X.shape[2])
    ManualRNN(model_config)

    return TrainingScaffold(
        train_shape=sequence_splits.train.X.shape,
        val_shape=sequence_splits.val.X.shape,
        test_shape=sequence_splits.test.X.shape,
    )

