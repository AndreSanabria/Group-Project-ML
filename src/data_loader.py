from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

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


def download_and_extract_csv(
    url: str,
    archive_path: Path,
    extracted_csv_path: Path,
    force: bool = False,
) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if extracted_csv_path.exists() and not force:
        return extracted_csv_path

    if force or not archive_path.exists():
        urlretrieve(url, archive_path)

    with zipfile.ZipFile(archive_path, "r") as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found in archive: {archive_path}")

        selected_member = next(
            (name for name in csv_members if Path(name).name == extracted_csv_path.name),
            csv_members[0],
        )
        with archive.open(selected_member, "r") as source, extracted_csv_path.open(
            "wb"
        ) as destination:
            shutil.copyfileobj(source, destination)

    return extracted_csv_path


def load_time_series_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}. "
            "Check data/README.md for the expected dataset location."
        )

    frame = pd.read_csv(csv_path, low_memory=False)
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
    frame.sort_index().to_csv(
        csv_path, index_label="timestamp", date_format="%Y-%m-%d %H:%M:%S"
    )
