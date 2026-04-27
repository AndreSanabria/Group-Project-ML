from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import run_persistence_baseline
from src.config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    ExperimentConfig,
    ensure_project_dirs,
)
from src.data_loader import load_time_series_csv, save_time_series_csv
from src.preprocessing import resample_to_hourly, select_feature_columns
from src.train_lstm import run_lstm_scaffold
from src.train_rnn import run_rnn_scaffold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffolded project runner for the Jena Climate RNN/LSTM project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Convert the raw Jena CSV into an hourly processed dataset."
    )
    preprocess_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the raw Jena CSV.",
    )
    preprocess_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Path to save the processed hourly CSV.",
    )
    preprocess_parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Columns to keep in the processed dataset.",
    )

    baseline_parser = subparsers.add_parser(
        "baseline", help="Run the persistence baseline on the hourly processed dataset."
    )
    baseline_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Path to the hourly processed CSV.",
    )
    baseline_parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Number of historical hourly steps in each input window.",
    )
    baseline_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in hours.",
    )
    baseline_parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column to predict.",
    )
    baseline_parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Feature columns to use for the experiment.",
    )

    rnn_parser = subparsers.add_parser(
        "train-rnn",
        help="Prepare normalized sequence splits and initialize the manual RNN scaffold.",
    )
    rnn_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Path to the hourly processed CSV.",
    )
    rnn_parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Number of historical hourly steps in each input window.",
    )
    rnn_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in hours.",
    )
    rnn_parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column to predict.",
    )
    rnn_parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Feature columns to use for the experiment.",
    )

    lstm_parser = subparsers.add_parser(
        "train-lstm",
        help="Prepare normalized sequence splits and initialize the manual LSTM scaffold.",
    )
    lstm_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Path to the hourly processed CSV.",
    )
    lstm_parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Number of historical hourly steps in each input window.",
    )
    lstm_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in hours.",
    )
    lstm_parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column to predict.",
    )
    lstm_parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Feature columns to use for the experiment.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace, input_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=input_path.name,
        target_column=args.target_column if hasattr(args, "target_column") else DEFAULT_TARGET_COLUMN,
        feature_columns=tuple(args.features) if hasattr(args, "features") else DEFAULT_FEATURE_COLUMNS,
        window_size=args.window_size if hasattr(args, "window_size") else 24,
        horizon=args.horizon if hasattr(args, "horizon") else 1,
    )


def run_preprocess(input_path: Path, output_path: Path, features: list[str]) -> None:
    frame = load_time_series_csv(input_path)
    hourly_frame = resample_to_hourly(frame)
    selected_frame = select_feature_columns(hourly_frame, tuple(features))
    save_time_series_csv(selected_frame, output_path)
    print(
        f"Saved hourly dataset to {output_path} "
        f"with shape {selected_frame.shape} and columns {list(selected_frame.columns)}."
    )


def run_baseline(input_path: Path, config: ExperimentConfig) -> None:
    hourly_frame = load_time_series_csv(input_path)
    artifacts = run_persistence_baseline(hourly_frame, config)
    test_metrics = artifacts.metrics_by_split["test"]
    print(
        f"Baseline complete. Test MAE={test_metrics['mae']:.4f}, "
        f"Test RMSE={test_metrics['rmse']:.4f}. Plot saved to {artifacts.plot_path}."
    )


def run_rnn(input_path: Path, config: ExperimentConfig) -> None:
    hourly_frame = load_time_series_csv(input_path)
    scaffold = run_rnn_scaffold(hourly_frame, config)
    print(
        f"RNN scaffold ready. Train shape={scaffold.train_shape}, "
        f"Val shape={scaffold.val_shape}, Test shape={scaffold.test_shape}. "
        f"Next step: implement ManualRNN.fit/predict in src/manual_rnn.py."
    )


def run_lstm(input_path: Path, config: ExperimentConfig) -> None:
    hourly_frame = load_time_series_csv(input_path)
    scaffold = run_lstm_scaffold(hourly_frame, config)
    print(
        f"LSTM scaffold ready. Train shape={scaffold.train_shape}, "
        f"Val shape={scaffold.val_shape}, Test shape={scaffold.test_shape}. "
        f"Next step: implement ManualLSTM.fit/predict in src/manual_lstm.py."
    )


def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    if args.command == "preprocess":
        run_preprocess(args.input, args.output, args.features)
        return

    config = build_config(args, args.input)
    if args.command == "baseline":
        run_baseline(args.input, config)
    elif args.command == "train-rnn":
        run_rnn(args.input, config)
    elif args.command == "train-lstm":
        run_lstm(args.input, config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
