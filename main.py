from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import run_persistence_baseline
from src.config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_PROCESSED_METADATA_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_RAW_ARCHIVE_PATH,
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_RESAMPLE_HOURS,
    DEFAULT_TARGET_COLUMN,
    ExperimentConfig,
    JENA_CLIMATE_DATA_URL,
    ensure_project_dirs,
)
from src.data_loader import download_and_extract_csv, load_time_series_csv, save_time_series_csv
from src.preprocessing import (
    preprocess_hourly_dataset,
    save_summary_json,
    summarize_time_series_frame,
)
from src.manual_lstm import ManualLSTMConfig
from src.manual_rnn import ManualRNNConfig
from src.train_lstm import run_lstm_training
from src.train_rnn import run_rnn_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffolded project runner for the Jena Climate RNN/LSTM project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-data",
        help="Download and extract the public Jena Climate dataset into data/raw/.",
    )
    download_parser.add_argument(
        "--url",
        default=JENA_CLIMATE_DATA_URL,
        help="Public URL for the Jena Climate ZIP archive.",
    )
    download_parser.add_argument(
        "--archive-output",
        type=Path,
        default=DEFAULT_RAW_ARCHIVE_PATH,
        help="Path to save the downloaded ZIP archive.",
    )
    download_parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to save the extracted CSV file.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the files already exist.",
    )

    describe_parser = subparsers.add_parser(
        "describe-data",
        help="Print a summary of a raw or processed time-series CSV and optionally save it.",
    )
    describe_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the CSV to summarize.",
    )
    describe_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the summary report.",
    )
    describe_parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional feature subset to summarize.",
    )

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
        "--summary-output",
        type=Path,
        default=DEFAULT_PROCESSED_METADATA_PATH,
        help="Path to save preprocessing metadata as JSON.",
    )
    preprocess_parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Columns to keep in the processed dataset.",
    )
    preprocess_parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_RESAMPLE_HOURS,
        help="Number of hours per resampled record.",
    )
    preprocess_parser.add_argument(
        "--aggregation",
        choices=("mean", "median"),
        default="mean",
        help="Aggregation to use when resampling the raw time series.",
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
    baseline_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological training split ratio.",
    )
    baseline_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Chronological validation split ratio.",
    )
    baseline_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Chronological test split ratio.",
    )

    rnn_parser = subparsers.add_parser(
        "train-rnn",
        help="Train and evaluate the manual RNN on the processed hourly dataset.",
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
    rnn_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological training split ratio.",
    )
    rnn_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Chronological validation split ratio.",
    )
    rnn_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Chronological test split ratio.",
    )
    rnn_parser.add_argument(
        "--hidden-size",
        type=int,
        default=16,
        help="Number of hidden units in the manual RNN.",
    )
    rnn_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for sample-wise SGD updates.",
    )
    rnn_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for the manual RNN.",
    )
    rnn_parser.add_argument(
        "--gradient-clip-value",
        type=float,
        default=1.0,
        help="Global gradient norm clipping value used during RNN training.",
    )
    rnn_parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20_000,
        help="Optional cap on samples seen per epoch. Use 0 to train on all samples.",
    )

    lstm_parser = subparsers.add_parser(
        "train-lstm",
        help="Train and evaluate the manual LSTM on the processed hourly dataset.",
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
    lstm_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological training split ratio.",
    )
    lstm_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Chronological validation split ratio.",
    )
    lstm_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Chronological test split ratio.",
    )
    lstm_parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Number of hidden units in the manual LSTM.",
    )
    lstm_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for sample-wise SGD updates.",
    )
    lstm_parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs for the manual LSTM.",
    )
    lstm_parser.add_argument(
        "--gradient-clip-value",
        type=float,
        default=5.0,
        help="Absolute gradient clipping value used during LSTM training.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace, input_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=input_path.name,
        target_column=(
            args.target_column if hasattr(args, "target_column") else DEFAULT_TARGET_COLUMN
        ),
        feature_columns=(
            tuple(args.features) if hasattr(args, "features") else DEFAULT_FEATURE_COLUMNS
        ),
        window_size=args.window_size if hasattr(args, "window_size") else 24,
        horizon=args.horizon if hasattr(args, "horizon") else 1,
        train_ratio=args.train_ratio if hasattr(args, "train_ratio") else 0.70,
        val_ratio=args.val_ratio if hasattr(args, "val_ratio") else 0.15,
        test_ratio=args.test_ratio if hasattr(args, "test_ratio") else 0.15,
        resample_hours=args.hours if hasattr(args, "hours") else DEFAULT_RESAMPLE_HOURS,
    )


def run_download_data(
    url: str, archive_output: Path, csv_output: Path, force: bool
) -> None:
    extracted_path = download_and_extract_csv(url, archive_output, csv_output, force=force)
    print(f"Dataset ready at {extracted_path}.")


def run_describe_data(
    input_path: Path, output_path: Path | None, features: list[str] | None
) -> None:
    frame = load_time_series_csv(input_path)
    if features:
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Requested summary features are missing: {missing}. "
                f"Available columns: {list(frame.columns)}"
            )
        frame = frame.loc[:, features]

    summary = summarize_time_series_frame(frame)
    if output_path is not None:
        save_summary_json(summary, output_path)

    print(
        f"Rows={summary['row_count']}, Columns={summary['column_count']}, "
        f"Start={summary['start_timestamp']}, End={summary['end_timestamp']}."
    )
    print(f"Features: {summary['columns']}")
    if output_path is not None:
        print(f"Summary saved to {output_path}.")


def run_preprocess(
    input_path: Path,
    output_path: Path,
    summary_output_path: Path,
    features: list[str],
    hours: int,
    aggregation: str,
) -> None:
    frame = load_time_series_csv(input_path)
    processed_frame, summary = preprocess_hourly_dataset(
        frame,
        tuple(features),
        resample_hours=hours,
        aggregation=aggregation,
    )
    save_time_series_csv(processed_frame, output_path)
    save_summary_json(summary, summary_output_path)
    print(
        f"Saved hourly dataset to {output_path} "
        f"with shape {processed_frame.shape} and columns {list(processed_frame.columns)}."
    )
    print(
        "Preprocessing summary: "
        f"{summary['raw_summary']['row_count']} raw rows -> "
        f"{summary['processed_summary']['row_count']} processed rows. "
        f"Metadata saved to {summary_output_path}."
    )


def run_baseline(input_path: Path, config: ExperimentConfig) -> None:
    hourly_frame = load_time_series_csv(input_path)
    artifacts = run_persistence_baseline(hourly_frame, config)
    test_metrics = artifacts.metrics_by_split["test"]
    print(
        f"Baseline complete. Test MAE={test_metrics['mae']:.4f}, "
        f"Test RMSE={test_metrics['rmse']:.4f}. Plot saved to {artifacts.plot_path}."
    )


def run_rnn(
    input_path: Path, config: ExperimentConfig, model_config: ManualRNNConfig
) -> None:
    hourly_frame = load_time_series_csv(input_path)
    artifacts = run_rnn_training(hourly_frame, config, model_config=model_config)
    test_metrics = artifacts.metrics_by_split["test"]
    val_metrics = artifacts.metrics_by_split["val"]
    print(
        f"RNN training complete. Train shape={artifacts.train_shape}, "
        f"Val shape={artifacts.val_shape}, Test shape={artifacts.test_shape}."
    )
    print(
        f"Validation MAE={val_metrics['mae']:.4f}, Validation RMSE={val_metrics['rmse']:.4f}, "
        f"Test MAE={test_metrics['mae']:.4f}, Test RMSE={test_metrics['rmse']:.4f}."
    )
    print(f"Predictions plot saved to {artifacts.plot_path}.")


def run_lstm(
    input_path: Path, config: ExperimentConfig, model_config: ManualLSTMConfig
) -> None:
    hourly_frame = load_time_series_csv(input_path)
    artifacts = run_lstm_training(hourly_frame, config, model_config=model_config)
    test_metrics = artifacts.metrics_by_split["test"]
    val_metrics = artifacts.metrics_by_split["val"]
    print(
        f"LSTM training complete. Train shape={artifacts.train_shape}, "
        f"Val shape={artifacts.val_shape}, Test shape={artifacts.test_shape}."
    )
    print(
        f"Validation MAE={val_metrics['mae']:.4f}, Validation RMSE={val_metrics['rmse']:.4f}, "
        f"Test MAE={test_metrics['mae']:.4f}, Test RMSE={test_metrics['rmse']:.4f}."
    )
    print(f"Predictions plot saved to {artifacts.plot_path}.")


def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    if args.command == "download-data":
        run_download_data(args.url, args.archive_output, args.csv_output, args.force)
        return

    if args.command == "describe-data":
        run_describe_data(args.input, args.output, args.features)
        return

    if args.command == "preprocess":
        run_preprocess(
            args.input,
            args.output,
            args.summary_output,
            args.features,
            args.hours,
            args.aggregation,
        )
        return

    config = build_config(args, args.input)
    if args.command == "baseline":
        run_baseline(args.input, config)
    elif args.command == "train-rnn":
        model_config = ManualRNNConfig(
            input_size=len(config.feature_columns),
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            random_seed=config.random_seed,
            gradient_clip=args.gradient_clip_value,
            max_train_samples=(
                args.max_train_samples if args.max_train_samples > 0 else None
            ),
        )
        run_rnn(args.input, config, model_config)
    elif args.command == "train-lstm":
        model_config = ManualLSTMConfig(
            input_size=len(config.feature_columns),
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            random_seed=config.random_seed,
            gradient_clip_value=args.gradient_clip_value,
        )
        run_lstm(args.input, config, model_config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
