from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_DIR = REPO_ROOT / "report"

DEFAULT_RAW_DATA_PATH = RAW_DATA_DIR / "jena_climate_2009_2016.csv"
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "jena_climate_hourly.csv"

DEFAULT_TARGET_COLUMN = "T (degC)"
DEFAULT_FEATURE_COLUMNS = (
    "T (degC)",
    "p (mbar)",
    "rh (%)",
    "wv (m/s)",
    "wd (deg)",
)


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


def ensure_project_dirs() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)

