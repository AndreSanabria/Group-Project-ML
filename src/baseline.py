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
from src.plots import plot_predictions, plot_residuals
from src.preprocessing import chronological_split, select_feature_columns
from src.sequences import SequenceDataset, create_sliding_window_dataset


@dataclass(slots=True)
class SplitEvaluation:
    dataset: SequenceDataset
    predictions: np.ndarray
    metrics: dict[str, float]


@dataclass(slots=True)
class BaselineArtifacts:
    metrics_by_split: dict[str, dict[str, float]]
    predictions_plot_path: Path
    residuals_plot_path: Path

    @property
    def plot_path(self) -> Path:
        return self.predictions_plot_path


def persistence_predict(dataset: SequenceDataset, target_index: int) -> np.ndarray:
    return dataset.X[:, -1, target_index]


def evaluate_split(
    frame: pd.DataFrame,
    target_column: str,
    window_size: int,
    horizon: int,
) -> SplitEvaluation:
    dataset = create_sliding_window_dataset(frame, target_column, window_size, horizon)
    target_index = frame.columns.get_loc(target_column)
    predictions = persistence_predict(dataset, target_index)
    metrics = regression_metrics(dataset.y, predictions)
    return SplitEvaluation(dataset=dataset, predictions=predictions, metrics=metrics)


def run_persistence_baseline(
    hourly_frame: pd.DataFrame,
    config: ExperimentConfig,
    metrics_summary_path: Path = RESULTS_DIR / "metrics_summary.csv",
    experiment_log_path: Path = RESULTS_DIR / "experiment_log.csv",
    predictions_plot_path: Path = RESULTS_DIR / "baseline_predictions.png",
    residuals_plot_path: Path = RESULTS_DIR / "baseline_residuals.png",
) -> BaselineArtifacts:
    selected_frame = select_feature_columns(hourly_frame, config.feature_columns)
    splits = chronological_split(
        selected_frame,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    split_results = {
        "train": evaluate_split(
            splits.train, config.target_column, config.window_size, config.horizon
        ),
        "val": evaluate_split(
            splits.val, config.target_column, config.window_size, config.horizon
        ),
        "test": evaluate_split(
            splits.test, config.target_column, config.window_size, config.horizon
        ),
    }

    for split_name, split_result in split_results.items():
        row = build_metrics_summary_row(
            run_name="persistence_baseline",
            model="PersistenceBaseline",
            split=split_name,
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            window_size=config.window_size,
            horizon=config.horizon,
            metrics=split_result.metrics,
        )
        append_csv_row(metrics_summary_path, row, METRICS_SUMMARY_FIELDS)

    experiment_row = build_experiment_log_row(
        experiment_number=next_experiment_number(experiment_log_path),
        model="PersistenceBaseline",
        dataset_name=config.dataset_name,
        target_column=config.target_column,
        feature_columns=config.feature_columns,
        window_size=config.window_size,
        horizon=config.horizon,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        mae_train=f"{split_results['train'].metrics['mae']:.6f}",
        rmse_train=f"{split_results['train'].metrics['rmse']:.6f}",
        mae_val=f"{split_results['val'].metrics['mae']:.6f}",
        rmse_val=f"{split_results['val'].metrics['rmse']:.6f}",
        mae_test=f"{split_results['test'].metrics['mae']:.6f}",
        rmse_test=f"{split_results['test'].metrics['rmse']:.6f}",
        plot_path=predictions_plot_path,
        notes=(
            "Persistence baseline: predict the next target value as the most recent "
            "target observed in the input window."
        ),
    )
    append_csv_row(experiment_log_path, experiment_row, EXPERIMENT_LOG_FIELDS)

    test_result = split_results["test"]
    plot_predictions(
        test_result.dataset.y,
        test_result.predictions,
        output_path=predictions_plot_path,
        title="Persistence Baseline: Actual vs Predicted Temperature",
    )
    plot_residuals(
        test_result.dataset.y,
        test_result.predictions,
        output_path=residuals_plot_path,
        title="Persistence Baseline: Test Residuals",
    )

    return BaselineArtifacts(
        metrics_by_split={name: result.metrics for name, result in split_results.items()},
        predictions_plot_path=predictions_plot_path,
        residuals_plot_path=residuals_plot_path,
    )

