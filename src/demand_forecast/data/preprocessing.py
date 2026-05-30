"""
demand_forecast/data/preprocessing.py
──────────────────────────────────────
Data ingestion and feature engineering for the demand forecasting pipeline.

MLOps KEY CONCEPTS demonstrated here:
  1. Data versioning  — every processed dataset gets a timestamp hash
  2. Schema validation — catch bad data early, before training
  3. Train/test split by TIME, never random (avoids leakage)
  4. Feature engineering inside a Pipeline object (prevents leakage)
  5. Artifact logging — track which data version trained which model
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Data schema — what we EXPECT the raw data to look like ───────────────────
# In production this would integrate with a data catalog (e.g. Dataplex on GCP)

REQUIRED_COLUMNS = {
    "week_start": "datetime64[ns]",
    "product_id": "object",
    "units_sold": "float64",
    "price": "float64",
    "promotion_flag": "int64",
    "store_count": "int64",
}

NUMERIC_FEATURES = ["price", "promotion_flag", "store_count"]


# ── Data classes for type-safe returns ───────────────────────────────────────

@dataclass
class DataSplit:
    """Holds train/val/test splits with metadata for MLflow logging."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    data_version: str            # hash for reproducibility
    split_date_val: str          # boundary date between train and val
    split_date_test: str         # boundary date between val and test
    stats: dict = field(default_factory=dict)


# ── Synthetic data generator (for local dev without real data) ────────────────

def generate_synthetic_sales_data(
    n_products: int = 10,
    n_weeks: int = 156,   # 3 years of weekly data
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic CPG sales data for development.

    In production this is replaced by a BigQuery query or GCS file read.
    Keeping it here lets us develop and test the full pipeline locally
    without needing real data or cloud access.
    """
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2021-01-04")   # Monday
    weeks = pd.date_range(start_date, periods=n_weeks, freq="W-MON")
    products = [f"PROD_{i:03d}" for i in range(n_products)]

    rows = []
    for product in products:
        # Each product has its own baseline and seasonality
        baseline = rng.uniform(500, 5000)
        trend = rng.uniform(-0.002, 0.005)   # slight growth or decline

        for i, week in enumerate(weeks):
            # Seasonality: peaks in Q4 (holiday), dip in Q1
            week_of_year = week.isocalendar().week
            seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * (week_of_year - 10) / 52)

            # Random promotions ~15% of weeks
            promotion = int(rng.random() < 0.15)
            price = rng.uniform(2.5, 15.0) * (0.85 if promotion else 1.0)

            units = (
                baseline
                * (1 + trend * i)
                * seasonality
                * (1.3 if promotion else 1.0)   # promo uplift
                * rng.lognormal(0, 0.1)          # noise
            )

            rows.append({
                "week_start": week,
                "product_id": product,
                "units_sold": max(0.0, round(units, 1)),
                "price": round(price, 2),
                "promotion_flag": promotion,
                "store_count": int(rng.uniform(50, 500)),
            })

    df = pd.DataFrame(rows).sort_values(["product_id", "week_start"]).reset_index(drop=True)
    logger.info(f"Generated synthetic data: {len(df):,} rows, {n_products} products, {n_weeks} weeks")
    return df


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """
    Enforce data contract before any processing.

    MLOPS INSIGHT: Data validation is the #1 defence against silent pipeline
    failures. A missing column discovered at training time costs hours.
    Discovered here it costs seconds.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col, dtype in REQUIRED_COLUMNS.items():
        if col == "week_start":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be datetime, got {df[col].dtype}")
        elif col in ("units_sold", "price"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}")

    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    neg_units = (df["units_sold"] < 0).sum()
    if neg_units > 0:
        raise ValueError(f"Found {neg_units} rows with negative units_sold")

    logger.info("Schema validation passed")


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    lag_windows: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """
    Create lag and rolling features for time series forecasting.

    MLOPS INSIGHT: Feature engineering logic MUST be identical at training
    and serving time. We encapsulate it here so both import the same function.
    Divergence between train and serve features = silent accuracy degradation.

    Args:
        df: Raw sales dataframe, sorted by (product_id, week_start)
        lag_windows: List of lag periods in weeks [1, 2, 4, ...]
        rolling_windows: List of rolling average windows in weeks [4, 8, 13]

    Returns:
        DataFrame with original columns plus engineered features.
        Rows with NaN features (start of each product's history) are dropped.
    """
    df = df.sort_values(["product_id", "week_start"]).copy()
    feature_cols = []

    # Group by product so lags don't bleed across product boundaries
    for product_id, group in df.groupby("product_id"):
        idx = group.index

        # Lag features: "what were sales N weeks ago?"
        for lag in lag_windows:
            col = f"lag_{lag}w"
            df.loc[idx, col] = group["units_sold"].shift(lag)
            feature_cols.append(col)

        # Rolling mean features: "what's the recent trend?"
        for window in rolling_windows:
            col = f"rolling_mean_{window}w"
            df.loc[idx, col] = (
                group["units_sold"].shift(1).rolling(window).mean()
            )
            # shift(1) ensures we don't use current week's sales as a feature
            feature_cols.append(col)

        # Rolling std: captures demand volatility
        col = f"rolling_std_4w"
        df.loc[idx, col] = group["units_sold"].shift(1).rolling(4).std()
        feature_cols.append(col)

    # Calendar features — encode seasonality
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["year"] = df["week_start"].dt.year

    # Sine/cosine encoding of week-of-year (avoids cliff at week 52 → 1)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    calendar_features = [
        "week_of_year", "month", "quarter", "year", "week_sin", "week_cos"
    ]

    # Drop rows where lags are NaN (unavoidable at start of history)
    all_feature_cols = list(dict.fromkeys(feature_cols)) + calendar_features
    df = df.dropna(subset=all_feature_cols).reset_index(drop=True)

    logger.info(
        f"Feature engineering complete: {len(df):,} rows, "
        f"{len(all_feature_cols)} engineered features"
    )
    return df, all_feature_cols


# ── Train / Val / Test split ──────────────────────────────────────────────────

def time_based_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_weeks: int,
    validation_weeks: int,
) -> DataSplit:
    """
    Split data chronologically into train / validation / test.

    CRITICAL MLOPS CONCEPT — WHY NOT RANDOM SPLIT?
    Time series data has temporal dependency — future values depend on past.
    A random split would let the model "see" future data during training
    (data leakage), producing optimistically biased evaluation metrics.
    Real performance is always measured on data AFTER the training period.

    Timeline:
    |──── train (bulk) ────|──── val (13w) ────|──── test (13w) ────|
                           ^                   ^
                    split_date_val       split_date_test
    """
    df = df.sort_values("week_start")
    all_weeks = df["week_start"].sort_values().unique()

    split_date_test = all_weeks[-(test_weeks)]
    split_date_val = all_weeks[-(test_weeks + validation_weeks)]

    train_mask = df["week_start"] < split_date_val
    val_mask = (df["week_start"] >= split_date_val) & (df["week_start"] < split_date_test)
    test_mask = df["week_start"] >= split_date_test

    # Compute a hash of the data for reproducibility tracking
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:8]
    data_version = f"v{datetime.now().strftime('%Y%m%d')}_{data_hash}"

    split = DataSplit(
        X_train=df[train_mask][feature_cols],
        y_train=df[train_mask][target_col],
        X_val=df[val_mask][feature_cols],
        y_val=df[val_mask][target_col],
        X_test=df[test_mask][feature_cols],
        y_test=df[test_mask][target_col],
        feature_names=feature_cols,
        data_version=data_version,
        split_date_val=str(split_date_val.date()),
        split_date_test=str(split_date_test.date()),
        stats={
            "train_rows": int(train_mask.sum()),
            "val_rows": int(val_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_period": f"{df[train_mask]['week_start'].min().date()} → {df[train_mask]['week_start'].max().date()}",
            "val_period": f"{df[val_mask]['week_start'].min().date()} → {df[val_mask]['week_start'].max().date()}",
            "test_period": f"{df[test_mask]['week_start'].min().date()} → {df[test_mask]['week_start'].max().date()}",
            "n_products": df["product_id"].nunique(),
        }
    )

    logger.info(
        f"Data split — train: {split.stats['train_rows']:,} | "
        f"val: {split.stats['val_rows']:,} | "
        f"test: {split.stats['test_rows']:,} | "
        f"version: {data_version}"
    )
    return split


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def run_preprocessing(
    lag_windows: list[int] = [1, 2, 4, 8, 12, 52],
    rolling_windows: list[int] = [4, 8, 13],
    test_weeks: int = 13,
    validation_weeks: int = 13,
    raw_df: pd.DataFrame | None = None,
) -> DataSplit:
    """
    Orchestrates the full preprocessing pipeline.

    In production:
      - raw_df comes from BigQuery or GCS
      - The DataSplit artifact is logged to MLflow
      - Processed features are written back to GCS for reuse

    For development, pass raw_df=None to use synthetic data.
    """
    if raw_df is None:
        logger.info("No raw data provided — generating synthetic data for development")
        raw_df = generate_synthetic_sales_data()

    # Ensure datetime type
    raw_df["week_start"] = pd.to_datetime(raw_df["week_start"])

    # Step 1: validate
    validate_schema(raw_df)

    # Step 2: engineer features
    df_features, feature_cols = engineer_features(raw_df, lag_windows, rolling_windows)

    # Step 3: include base numeric features
    all_features = NUMERIC_FEATURES + feature_cols

    # Step 4: time-based split
    split = time_based_split(
        df=df_features,
        target_col="units_sold",
        feature_cols=all_features,
        test_weeks=test_weeks,
        validation_weeks=validation_weeks,
    )

    return split
