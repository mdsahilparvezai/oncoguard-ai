"""
Data module for ONCOGUARD AI - Wisconsin Diagnostic Breast Cancer dataset handling.

This module provides:
- Dataset loading from UCI repository with schema validation
- Data quality checks and exploratory analysis
- Leakage-resistant preprocessing pipelines
- Train/test splits with stratification
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "wdbc.data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "wdbc_processed.parquet"

# Column names per UCI documentation
COLUMN_NAMES = [
    "id",
    "diagnosis",
    *[
        f"{feat}_{stat}"
        for feat in [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
        ]
        for stat in ["mean", "se", "worst"]
    ]
]

FEATURE_COLUMNS = COLUMN_NAMES[2:]  # Exclude id and diagnosis
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"

# Target mapping: M (malignant) = 1 (positive class), B (benign) = 0
TARGET_MAPPING = {"M": 1, "B": 0}
TARGET_NAMES = {0: "Benign", 1: "Malignant"}


@dataclass
class DatasetInfo:
    """Container for dataset metadata and validation results."""
    n_samples: int
    n_features: int
    class_distribution: Dict[int, int]
    missing_values: int
    duplicate_rows: int
    feature_stats: pd.DataFrame
    target_mapping: Dict[str, int]
    checksum: str


@dataclass
class SplitData:
    """Container for train/validation/test splits."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    target_names: Dict[int, str]
    scaler: Optional[StandardScaler] = None


def compute_checksum(df: pd.DataFrame) -> str:
    """Compute SHA256 checksum of dataframe for reproducibility verification."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataset schema against expected structure.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check column count
    expected_cols = len(COLUMN_NAMES)
    if len(df.columns) != expected_cols:
        errors.append(f"Expected {expected_cols} columns, got {len(df.columns)}")

    # Check column names if they match
    if list(df.columns) != COLUMN_NAMES:
        errors.append("Column names do not match expected schema")

    # Check target values
    unique_targets = set(df[TARGET_COLUMN].unique())
    expected_targets = set(TARGET_MAPPING.keys())
    if unique_targets != expected_targets:
        errors.append(f"Target values mismatch. Expected {expected_targets}, got {unique_targets}")

    # Check for missing values
    if df.isnull().any().any():
        errors.append("Dataset contains missing values")

    # Check ID column uniqueness
    if df[ID_COLUMN].duplicated().any():
        errors.append("ID column contains duplicate values")

    # Check feature columns are numeric
    feature_cols = df[FEATURE_COLUMNS].select_dtypes(include=[np.number]).columns
    if len(feature_cols) != len(FEATURE_COLUMNS):
        non_numeric = set(FEATURE_COLUMNS) - set(feature_cols)
        errors.append(f"Non-numeric feature columns: {non_numeric}")

    return len(errors) == 0, errors


def _load_wdbc_file(path_or_url: str, is_url: bool = False) -> pd.DataFrame:
    """
    Load WDBC file, handling the optional header row.

    The UCI wdbc.data file has NO header row, but the wdbc.data.txt
    and wdbc.names files do. We try header=None first and fall back
    to header=0 if the first row looks like a header.
    """
    if is_url:
        df = pd.read_csv(path_or_url, header=None, names=COLUMN_NAMES)
    else:
        df = pd.read_csv(path_or_url, header=None, names=COLUMN_NAMES)

    # Check if first row is actually a header (contains 'diagnosis' in target column)
    if df[TARGET_COLUMN].iloc[0] == "diagnosis":
        df = df.iloc[1:].reset_index(drop=True)

    # Convert feature columns to numeric (coerce errors to NaN)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows with NaN (from header row or bad data)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load raw Wisconsin Diagnostic Breast Cancer dataset.

    Args:
        force_download: If True, re-download from UCI even if local copy exists.

    Returns:
        Raw DataFrame with validated schema.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force_download:
        logger.info(f"Loading dataset from local cache: {RAW_DATA_PATH}")
        df = _load_wdbc_file(RAW_DATA_PATH)
    else:
        logger.info(f"Downloading dataset from UCI: {UCI_URL}")
        try:
            df = _load_wdbc_file(UCI_URL, is_url=True)
            df.to_csv(RAW_DATA_PATH, index=False)
            logger.info(f"Saved raw dataset to: {RAW_DATA_PATH}")
        except Exception as e:
            if RAW_DATA_PATH.exists():
                logger.warning(f"Download failed, using cached version: {e}")
                df = _load_wdbc_file(RAW_DATA_PATH)
            else:
                raise RuntimeError(f"Failed to load dataset from UCI and no local cache: {e}")

    # Validate schema
    is_valid, errors = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

    logger.info(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def analyze_data_quality(df: pd.DataFrame) -> DatasetInfo:
    """
    Perform comprehensive data quality analysis.

    Returns:
        DatasetInfo with quality metrics.
    """
    # Class distribution
    class_dist = df[TARGET_COLUMN].map(TARGET_MAPPING).value_counts().to_dict()

    # Feature statistics
    feature_stats = df[FEATURE_COLUMNS].describe().T
    feature_stats["missing"] = df[FEATURE_COLUMNS].isnull().sum()
    feature_stats["zeros"] = (df[FEATURE_COLUMNS] == 0).sum()

    info = DatasetInfo(
        n_samples=len(df),
        n_features=len(FEATURE_COLUMNS),
        class_distribution=class_dist,
        missing_values=int(df.isnull().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        feature_stats=feature_stats,
        target_mapping=TARGET_MAPPING,
        checksum=compute_checksum(df)
    )

    logger.info(f"Data quality: {info.n_samples} samples, {info.n_features} features")
    logger.info(f"Class distribution: {info.class_distribution}")
    logger.info(f"Missing values: {info.missing_values}")
    logger.info(f"Duplicate rows: {info.duplicate_rows}")
    logger.info(f"Checksum: {info.checksum[:16]}...")

    return info


def create_preprocessing_pipeline() -> Pipeline:
    """
    Create a leakage-resistant preprocessing pipeline.

    The pipeline:
    1. Drops ID column
    2. Maps target to binary (0=Benign, 1=Malignant)
    3. Standardizes features (fit only on training data)

    Returns:
        Sklearn Pipeline for preprocessing.
    """
    # Feature preprocessing: standardize all features
    feature_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS)
        ],
        remainder="drop"  # Drops ID column automatically
    )

    pipeline = Pipeline([
        ("features", feature_transformer)
    ])

    return pipeline


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Args:
        df: Raw dataframe with all columns.

    Returns:
        Tuple of (X, y) where X excludes ID and target, y is binary encoded.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)

    # Verify no leakage
    assert ID_COLUMN not in X.columns, "ID column leaked into features"
    assert TARGET_COLUMN not in X.columns, "Target column leaked into features"
    assert y.isin([0, 1]).all(), "Target contains invalid values"

    return X, y


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> SplitData:
    """
    Create train/validation/test splits with stratification.

    Args:
        df: Full dataset.
        test_size: Fraction for test set.
        val_size: Fraction of remaining data for validation.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify by target.

    Returns:
        SplitData with all splits and metadata.
    """
    X, y = prepare_features_target(df)

    stratify_arg = y if stratify else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    stratify_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )

    # Fit scaler on training data only (leakage prevention)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=FEATURE_COLUMNS,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS,
        index=X_test.index
    )

    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train class dist: {y_train.value_counts().to_dict()}")
    logger.info(f"Val class dist: {y_val.value_counts().to_dict()}")
    logger.info(f"Test class dist: {y_test.value_counts().to_dict()}")

    return SplitData(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=FEATURE_COLUMNS,
        target_names=TARGET_NAMES,
        scaler=scaler
    )


def save_processed_data(split_data: SplitData, path: Path = PROCESSED_DATA_PATH) -> None:
    """Save processed splits to disk for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet for efficiency
    split_data.X_train.to_parquet(path.parent / "X_train.parquet")
    split_data.X_val.to_parquet(path.parent / "X_val.parquet")
    split_data.X_test.to_parquet(path.parent / "X_test.parquet")
    split_data.y_train.to_frame("target").to_parquet(path.parent / "y_train.parquet")
    split_data.y_val.to_frame("target").to_parquet(path.parent / "y_val.parquet")
    split_data.y_test.to_frame("target").to_parquet(path.parent / "y_test.parquet")

    # Save metadata
    import json
    metadata = {
        "feature_names": split_data.feature_names,
        "target_names": split_data.target_names,
        "target_mapping": TARGET_MAPPING,
        "n_train": len(split_data.X_train),
        "n_val": len(split_data.X_val),
        "n_test": len(split_data.X_test)
    }
    with open(path.parent / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save scaler
    joblib.dump(split_data.scaler, path.parent / "scaler.joblib")

    logger.info(f"Saved processed data to: {path.parent}")


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> SplitData:
    """Load processed splits from disk."""
    import json

    X_train = pd.read_parquet(path.parent / "X_train.parquet")
    X_val = pd.read_parquet(path.parent / "X_val.parquet")
    X_test = pd.read_parquet(path.parent / "X_test.parquet")
    y_train = pd.read_parquet(path.parent / "y_train.parquet")["target"]
    y_val = pd.read_parquet(path.parent / "y_val.parquet")["target"]
    y_test = pd.read_parquet(path.parent / "y_test.parquet")["target"]
    scaler = joblib.load(path.parent / "scaler.joblib")

    with open(path.parent / "metadata.json", "r") as f:
        metadata = json.load(f)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=metadata["feature_names"],
        target_names=metadata["target_names"],
        scaler=scaler
    )


def get_feature_definitions() -> Dict[str, str]:
    """Return human-readable definitions for all features."""
    base_definitions = {
        "radius": "Mean distance from center to points on the perimeter",
        "texture": "Standard deviation of gray-scale values (texture)",
        "perimeter": "Perimeter of the nucleus",
        "area": "Area of the nucleus",
        "smoothness": "Local variation in radius lengths",
        "compactness": "Perimeter^2 / area - 1.0",
        "concavity": "Severity of concave portions of the contour",
        "concave_points": "Number of concave portions of the contour",
        "symmetry": "Symmetry of the nucleus",
        "fractal_dimension": "Coastline approximation - 1"
    }

    definitions = {}
    for feat in ["radius", "texture", "perimeter", "area", "smoothness",
                 "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]:
        for stat in ["mean", "se", "worst"]:
            col = f"{feat}_{stat}"
            stat_desc = {"mean": "Mean", "se": "Standard Error", "worst": "Worst (mean of 3 largest)"}[stat]
            definitions[col] = f"{stat_desc} {base_definitions[feat].lower()}"

    return definitions


# Convenience function for full pipeline
def run_data_pipeline(force_download: bool = False) -> Tuple[SplitData, DatasetInfo]:
    """
    Run complete data pipeline: load, validate, analyze, split, save.

    Returns:
        Tuple of (SplitData, DatasetInfo)
    """
    logger.info("Starting data pipeline...")

    # Load and validate
    df = load_raw_data(force_download=force_download)

    # Analyze
    info = analyze_data_quality(df)

    # Split
    splits = create_splits(df)

    # Save
    save_processed_data(splits)

    logger.info("Data pipeline completed successfully")
    return splits, info


if __name__ == "__main__":
    # Quick test when run directly
    splits, info = run_data_pipeline()
    print(f"Pipeline test passed: {splits.X_train.shape}, {splits.y_train.shape}")