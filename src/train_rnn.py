from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig, RESULTS_DIR
from src.evaluate import (
    EXPERIMENT_LOG_FIELDS,
    METRICS_SUMMARY_FIELDS,
    append_csv_row,
    build_experiment_log_row,
    build_metrics_summary_row,
    next_experiment_number,
    regression_metrics,
)
from src.manual_rnn import ManualRNN, ManualRNNConfig
from src.plots import plot_predictions
from src.preprocessing import prepare_normalized_sequence_splits


@dataclass(slots=True)
class TrainingScaffold:
    train_shape: tuple[int, ...]
    val_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    metrics_by_split: dict[str, dict[str, float]]
    training_history: dict[str, list[float]]
    plot_path: Path


def _denormalize_target(
    values: np.ndarray, target_mean: float, target_std: float
) -> np.ndarray:
    return (values * target_std) + target_mean


def run_rnn_scaffold(
    hourly_frame: pd.DataFrame,
    config: ExperimentConfig,
    metrics_summary_path: Path = RESULTS_DIR / "metrics_summary.csv",
    experiment_log_path: Path = RESULTS_DIR / "experiment_log.csv",
    plot_path: Path = RESULTS_DIR / "rnn_predictions.png",
) -> TrainingScaffold:
    sequence_splits = prepare_normalized_sequence_splits(hourly_frame, config)
    model_config = ManualRNNConfig(
        input_size=sequence_splits.train.X.shape[2],
        hidden_size=16,
        learning_rate=1e-3,
        epochs=10,
        random_seed=config.random_seed,
        gradient_clip=1.0,
        max_train_samples=20_000,
    )
    model = ManualRNN(model_config)
    training_history = model.fit(sequence_splits.train.X, sequence_splits.train.y)

    target_mean = float(sequence_splits.normalization_stats.means[config.target_column])
    target_std = float(sequence_splits.normalization_stats.stds[config.target_column])

    split_predictions = {
        "val": model.predict(sequence_splits.val.X),
        "test": model.predict(sequence_splits.test.X),
    }
    split_targets = {
        "val": sequence_splits.val.y,
        "test": sequence_splits.test.y,
    }

    metrics_by_split: dict[str, dict[str, float]] = {}
    denormalized_predictions: dict[str, np.ndarray] = {}
    denormalized_targets: dict[str, np.ndarray] = {}

    for split_name in ("val", "test"):
        denormalized_predictions[split_name] = _denormalize_target(
            split_predictions[split_name], target_mean, target_std
        )
        denormalized_targets[split_name] = _denormalize_target(
            split_targets[split_name], target_mean, target_std
        )
        metrics_by_split[split_name] = regression_metrics(
            denormalized_targets[split_name], denormalized_predictions[split_name]
        )

        row = build_metrics_summary_row(
            run_name="manual_rnn",
            model="ManualRNN",
            split=split_name,
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            window_size=config.window_size,
            horizon=config.horizon,
            metrics=metrics_by_split[split_name],
        )
        append_csv_row(metrics_summary_path, row, METRICS_SUMMARY_FIELDS)

    final_loss = training_history["loss"][-1] if training_history["loss"] else float("nan")
    experiment_row = build_experiment_log_row(
        experiment_number=next_experiment_number(experiment_log_path),
        model="ManualRNN",
        dataset_name=config.dataset_name,
        target_column=config.target_column,
        feature_columns=config.feature_columns,
        window_size=config.window_size,
        horizon=config.horizon,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        hidden_size=model_config.hidden_size,
        learning_rate=model_config.learning_rate,
        epochs=model_config.epochs,
        mae_val=f"{metrics_by_split['val']['mae']:.6f}",
        rmse_val=f"{metrics_by_split['val']['rmse']:.6f}",
        mae_test=f"{metrics_by_split['test']['mae']:.6f}",
        rmse_test=f"{metrics_by_split['test']['rmse']:.6f}",
        notes=(
            "Manual tanh RNN with full BPTT, clipped SGD, "
            f"max_train_samples={model_config.max_train_samples}, "
            f"final_normalized_mse={final_loss:.6f}."
        ),
    )
    append_csv_row(experiment_log_path, experiment_row, EXPERIMENT_LOG_FIELDS)

    plot_predictions(
        denormalized_targets["test"],
        denormalized_predictions["test"],
        output_path=plot_path,
        title="Manual RNN: Actual vs Predicted Temperature",
    )

    return TrainingScaffold(
        train_shape=sequence_splits.train.X.shape,
        val_shape=sequence_splits.val.X.shape,
        test_shape=sequence_splits.test.X.shape,
        metrics_by_split=metrics_by_split,
        training_history=training_history,
        plot_path=plot_path,
    )

