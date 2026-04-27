from __future__ import annotations

from pathlib import Path

import pandas as pd


TIMESTAMP_COLUMN_CANDIDATES = (
    "Date Time",
    "timestamp",
    "Timestamp",
    "datetime",
    "Datetime",
    "date",
    "Date",
)


def _detect_timestamp_column(columns: list[str]) -> str:
    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a timestamp column. "
        f"Expected one of: {TIMESTAMP_COLUMN_CANDIDATES}"
    )


def _parse_timestamp_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    if parsed.notna().sum() >= int(len(series) * 0.90):
        return parsed
    return pd.to_datetime(series, errors="coerce")


def load_time_series_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}. "
            "Check data/README.md for the expected dataset location."
        )

    frame = pd.read_csv(csv_path)
    timestamp_column = _detect_timestamp_column(frame.columns.tolist())
    timestamps = _parse_timestamp_column(frame[timestamp_column])

    parsed_frame = frame.drop(columns=[timestamp_column]).copy()
    for column in parsed_frame.columns:
        parsed_frame[column] = pd.to_numeric(parsed_frame[column], errors="coerce")

    parsed_frame.insert(0, "timestamp", timestamps)
    parsed_frame = parsed_frame.dropna(subset=["timestamp"])
    parsed_frame = parsed_frame.set_index("timestamp").sort_index()
    parsed_frame = parsed_frame[~parsed_frame.index.duplicated(keep="first")]

    numeric_frame = parsed_frame.select_dtypes(include=["number"]).dropna(how="all")
    if numeric_frame.empty:
        raise ValueError(f"No numeric feature columns were found in {csv_path}.")

    return numeric_frame


def save_time_series_csv(frame: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index_label="timestamp")

