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
    format_path_for_log,
    next_experiment_number,
    regression_metrics,
)
from src.manual_lstm import ManualLSTM, ManualLSTMConfig
from src.plots import plot_predictions, plot_residuals
from src.preprocessing import prepare_normalized_sequence_splits


@dataclass(slots=True)
class LSTMTrainingArtifacts:
    train_shape: tuple[int, ...]
    val_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    history: dict[str, list[float]]
    metrics_by_split: dict[str, dict[str, float]]
    predictions_plot_path: Path
    residuals_plot_path: Path

    @property
    def plot_path(self) -> Path:
        return self.predictions_plot_path


def _restore_target_scale(
    values: np.ndarray, config: ExperimentConfig, target_mean: float, target_std: float
) -> np.ndarray:
    restored_values = np.asarray(values, dtype=float).reshape(-1) * target_std + target_mean
    if config.target_column not in config.feature_columns:
        raise ValueError(
            f"Target column '{config.target_column}' must be included in feature_columns."
        )
    return restored_values


def run_lstm_training(
    hourly_frame: pd.DataFrame,
    config: ExperimentConfig,
    model_config: ManualLSTMConfig | None = None,
    metrics_summary_path: Path = RESULTS_DIR / "metrics_summary.csv",
    experiment_log_path: Path = RESULTS_DIR / "experiment_log.csv",
    predictions_plot_path: Path = RESULTS_DIR / "lstm_predictions.png",
    residuals_plot_path: Path = RESULTS_DIR / "lstm_residuals.png",
) -> LSTMTrainingArtifacts:
    sequence_splits = prepare_normalized_sequence_splits(hourly_frame, config)
    if model_config is None:
        model_config = ManualLSTMConfig(
            input_size=sequence_splits.train.X.shape[2],
            random_seed=config.random_seed,
        )

    model = ManualLSTM(model_config)
    history = model.fit(sequence_splits.train.X, sequence_splits.train.y)

    target_mean = float(sequence_splits.normalization_stats.means[config.target_column])
    target_std = float(sequence_splits.normalization_stats.stds[config.target_column])

    metrics_by_split: dict[str, dict[str, float]] = {}
    predictions_by_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    datasets_by_split = {
        "train": sequence_splits.train,
        "val": sequence_splits.val,
        "test": sequence_splits.test,
    }

    for split_name, dataset in datasets_by_split.items():
        predictions = model.predict(dataset.X)
        true_values = _restore_target_scale(dataset.y, config, target_mean, target_std)
        predicted_values = _restore_target_scale(predictions, config, target_mean, target_std)
        metrics = regression_metrics(true_values, predicted_values)

        metrics_by_split[split_name] = metrics
        predictions_by_split[split_name] = (true_values, predicted_values)

        summary_row = build_metrics_summary_row(
            run_name="manual_lstm",
            model="ManualLSTM",
            split=split_name,
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            window_size=config.window_size,
            horizon=config.horizon,
            metrics=metrics,
        )
        append_csv_row(metrics_summary_path, summary_row, METRICS_SUMMARY_FIELDS)

    experiment_row = build_experiment_log_row(
        experiment_number=next_experiment_number(experiment_log_path),
        model="ManualLSTM",
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
        mae_train=f"{metrics_by_split['train']['mae']:.6f}",
        rmse_train=f"{metrics_by_split['train']['rmse']:.6f}",
        mae_val=f"{metrics_by_split['val']['mae']:.6f}",
        rmse_val=f"{metrics_by_split['val']['rmse']:.6f}",
        mae_test=f"{metrics_by_split['test']['mae']:.6f}",
        rmse_test=f"{metrics_by_split['test']['rmse']:.6f}",
        plot_path=format_path_for_log(predictions_plot_path),
        notes=(
            "Manual many-to-one LSTM regressor trained with sample-wise SGD and "
            "backpropagation through time. "
            f"max_train_samples={model_config.max_train_samples}."
        ),
    )
    append_csv_row(experiment_log_path, experiment_row, EXPERIMENT_LOG_FIELDS)

    test_true_values, test_predicted_values = predictions_by_split["test"]
    plot_predictions(
        test_true_values,
        test_predicted_values,
        output_path=predictions_plot_path,
        title="Manual LSTM: Actual vs Predicted Temperature",
    )
    plot_residuals(
        test_true_values,
        test_predicted_values,
        output_path=residuals_plot_path,
        title="Manual LSTM: Test Residuals",
    )

    return LSTMTrainingArtifacts(
        train_shape=sequence_splits.train.X.shape,
        val_shape=sequence_splits.val.X.shape,
        test_shape=sequence_splits.test.X.shape,
        history=history,
        metrics_by_split=metrics_by_split,
        predictions_plot_path=predictions_plot_path,
        residuals_plot_path=residuals_plot_path,
    )
