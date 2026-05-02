from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_DIR = REPO_ROOT / "report"

JENA_CLIMATE_DATA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

DEFAULT_RAW_ARCHIVE_PATH = RAW_DATA_DIR / "jena_climate_2009_2016.csv.zip"
DEFAULT_RAW_DATA_PATH = RAW_DATA_DIR / "jena_climate_2009_2016.csv"
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "jena_climate_hourly.csv"
DEFAULT_PROCESSED_METADATA_PATH = PROCESSED_DATA_DIR / "jena_climate_hourly_metadata.json"

DEFAULT_TARGET_COLUMN = "T (degC)"
DEFAULT_FEATURE_COLUMNS = (
    "T (degC)",
    "p (mbar)",
    "rh (%)",
    "wv (m/s)",
    "wd (deg)",
)
DEFAULT_RESAMPLE_HOURS = 1

# The TensorFlow Jena Climate tutorial flags these as erroneous wind values.
DEFAULT_SENTINEL_REPLACEMENTS = {
    "wv (m/s)": (-9999.0,),
    "max. wv (m/s)": (-9999.0,),
}

@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str = DEFAULT_RAW_DATA_PATH.name
    target_column: str = DEFAULT_TARGET_COLUMN
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    window_size: int = 24
    horizon: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    resample_hours: int = DEFAULT_RESAMPLE_HOURS

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.horizon <= 0:
            raise ValueError("window_size and horizon must both be positive integers.")
        if self.resample_hours <= 0:
            raise ValueError("resample_hours must be a positive integer.")
        if self.target_column not in self.feature_columns:
            raise ValueError(
                f"Target column '{self.target_column}' must be included in feature_columns."
            )
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError("Train/validation/test ratios must sum to 1.0.")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("Train/validation/test ratios must all be positive.")


def ensure_project_dirs() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
