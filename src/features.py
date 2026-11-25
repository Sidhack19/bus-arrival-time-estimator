"""Feature parsing, data cleaning, and sklearn preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import (
    MAX_ARRIVAL_TIME_MIN,
    MAX_DISTANCE_KM,
    MAX_PREV_TIME_MIN,
    MIN_ARRIVAL_TIME_MIN,
    MIN_DISTANCE_KM,
    MIN_PREV_TIME_MIN,
    MODEL_INPUT_COLUMNS,
    TARGET_COLUMN,
    TRAFFIC_CATEGORIES,
)


@dataclass
class DataQualityReport:
    """Summary of raw dataset checks before model training."""

    total_rows: int
    missing_target_rows: int
    invalid_distance_rows: int
    invalid_time_rows: int
    invalid_prev_time_rows: int
    invalid_target_rows: int
    invalid_traffic_rows: int
    duplicate_rows: int
    cleaned_rows: int
    missing_feature_counts: dict[str, int]


def parse_time_to_hours(value: Any) -> float:
    """Convert 'HH:MM' strings or numeric hours into decimal hours in [0, 24).

    Returns NaN when the input is missing or unparseable.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")

    if isinstance(value, (int, float, np.integer, np.floating)):
        hours = float(value)
        return hours if 0.0 <= hours < 24.0 else float("nan")

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return float("nan")

    if ":" in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            return float("nan")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
        except ValueError:
            return float("nan")
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            return float("nan")
        return hour + minute / 60.0 + second / 3600.0

    try:
        hours = float(text)
    except ValueError:
        return float("nan")
    return hours if 0.0 <= hours < 24.0 else float("nan")


def format_hours_to_hhmm(hours: float) -> str:
    """Format decimal hours in [0, 24) to 'HH:MM' 24-hour clock string."""
    total_minutes = int(round((hours % 24.0) * 60)) % (24 * 60)
    hh, mm = divmod(total_minutes, 60)
    return f"{hh:02d}:{mm:02d}"


def is_campus_rush_hour(hours: float) -> bool:
    """Return True if decimal hour falls within campus peak boarding windows."""
    if np.isnan(hours):
        return False
    return (
        (8.0 <= hours <= 9.75)
        or (12.25 <= hours <= 14.0)
        or (16.5 <= hours <= 18.25)
    )


class BusFeatureEngineer(BaseEstimator, TransformerMixin):
    """Expand raw 4-column bus inputs into numeric + ordinal ML features.

    Expects a DataFrame containing:
    - route_distance_km
    - time_of_day ('HH:MM' or decimal hour)
    - traffic_category ('Low', 'Medium', 'High')
    - previous_travel_time_min
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BusFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        hours = df["time_of_day"].apply(parse_time_to_hours).astype(float)
        radians = 2.0 * np.pi * (hours / 24.0)

        out = pd.DataFrame(index=df.index)
        out["route_distance_km"] = pd.to_numeric(
            df["route_distance_km"], errors="coerce"
        )
        out["previous_travel_time_min"] = pd.to_numeric(
            df["previous_travel_time_min"], errors="coerce"
        )
        out["hour_decimal"] = hours
        out["time_sin"] = np.sin(radians)
        out["time_cos"] = np.cos(radians)
        out["is_peak_hour"] = hours.apply(
            lambda h: np.nan if np.isnan(h) else float(is_campus_rush_hour(h))
        )

        traffic = df["traffic_category"].astype("object")
        valid_set = set(TRAFFIC_CATEGORIES)
        out["traffic_category"] = traffic.apply(
            lambda v: v if isinstance(v, str) and v in valid_set else np.nan
        )
        return out


def build_preprocessor() -> Pipeline:
    """Create the unfitted sklearn preprocessing pipeline.

    Important: Always fit this pipeline ONLY on the training split to prevent
    data leakage into validation or test sets.
    """
    numeric_features = [
        "route_distance_km",
        "previous_travel_time_min",
        "hour_decimal",
        "time_sin",
        "time_cos",
        "is_peak_hour",
    ]
    ordinal_features = ["traffic_category"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[TRAFFIC_CATEGORIES],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("ord", ordinal_pipeline, ordinal_features),
        ]
    )

    return Pipeline(
        steps=[
            ("engineer", BusFeatureEngineer()),
            ("columns", column_transformer),
        ]
    )


def get_transformed_feature_names() -> list[str]:
    """Return human-readable feature names matching build_preprocessor output."""
    return [
        "route_distance_km",
        "previous_travel_time_min",
        "hour_decimal",
        "time_sin",
        "time_cos",
        "is_peak_hour",
        "traffic_category_ord",
    ]


def clean_raw_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Audit raw rows, remove corrupted/targetless records, preserve imputable gaps.

    Missing values in `previous_travel_time_min` or `traffic_category` are retained
    so the train-only imputer handles them without leaking test statistics.
    """
    df = raw_df.copy()
    total_rows = len(df)

    # Standardize string traffic values while preserving missing values
    if "traffic_category" in df.columns:
        df["traffic_category"] = df["traffic_category"].apply(
            lambda v: v.strip().title() if isinstance(v, str) and v.strip() else np.nan
        )

    missing_target_mask = df[TARGET_COLUMN].isna()
    dist_num = pd.to_numeric(df["route_distance_km"], errors="coerce")
    invalid_distance_mask = dist_num.notna() & (
        (dist_num < MIN_DISTANCE_KM) | (dist_num > MAX_DISTANCE_KM)
    )
    # Missing distance cannot reliably locate a route; drop missing or impossible distances
    invalid_distance_mask = invalid_distance_mask | dist_num.isna()

    parsed_hours = df["time_of_day"].apply(parse_time_to_hours)
    invalid_time_mask = parsed_hours.isna()

    prev_num = pd.to_numeric(df["previous_travel_time_min"], errors="coerce")
    invalid_prev_mask = prev_num.notna() & (
        (prev_num < MIN_PREV_TIME_MIN) | (prev_num > MAX_PREV_TIME_MIN)
    )

    target_num = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    invalid_target_mask = (~missing_target_mask) & (
        target_num.isna()
        | (target_num < MIN_ARRIVAL_TIME_MIN)
        | (target_num > MAX_ARRIVAL_TIME_MIN)
    )

    valid_traffic = set(TRAFFIC_CATEGORIES)
    invalid_traffic_mask = df["traffic_category"].notna() & (
        ~df["traffic_category"].isin(valid_traffic)
    )
    # Convert unknown traffic labels to NaN so imputer can handle them if needed,
    # or flag corrupted rows explicitly. Here we treat non-domain labels as invalid rows.

    drop_mask = (
        missing_target_mask
        | invalid_distance_mask
        | invalid_time_mask
        | invalid_prev_mask
        | invalid_target_mask
        | invalid_traffic_mask
    )

    cleaned = df.loc[~drop_mask].copy()
    cleaned["route_distance_km"] = pd.to_numeric(cleaned["route_distance_km"])
    cleaned["previous_travel_time_min"] = pd.to_numeric(
        cleaned["previous_travel_time_min"], errors="coerce"
    )
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN])

    duplicate_count = int(cleaned.duplicated(subset=MODEL_INPUT_COLUMNS + [TARGET_COLUMN]).sum())
    if duplicate_count > 0:
        cleaned = cleaned.drop_duplicates(
            subset=MODEL_INPUT_COLUMNS + [TARGET_COLUMN]
        ).reset_index(drop=True)
    else:
        cleaned = cleaned.reset_index(drop=True)

    missing_counts = {
        col: int(cleaned[col].isna().sum()) for col in MODEL_INPUT_COLUMNS + [TARGET_COLUMN]
    }

    report = DataQualityReport(
        total_rows=total_rows,
        missing_target_rows=int(missing_target_mask.sum()),
        invalid_distance_rows=int(invalid_distance_mask.sum()),
        invalid_time_rows=int(invalid_time_mask.sum()),
        invalid_prev_time_rows=int(invalid_prev_mask.sum()),
        invalid_target_rows=int(invalid_target_mask.sum()),
        invalid_traffic_rows=int(invalid_traffic_mask.sum()),
        duplicate_rows=duplicate_count,
        cleaned_rows=len(cleaned),
        missing_feature_counts=missing_counts,
    )
    return cleaned, report


def validate_prediction_inputs(
    route_distance_km: float,
    time_of_day: str | float,
    traffic_category: str,
    previous_travel_time_min: float,
) -> pd.DataFrame:
    """Validate single-trip user inputs and return a 1-row DataFrame for inference."""
    if not (MIN_DISTANCE_KM <= float(route_distance_km) <= MAX_DISTANCE_KM):
        raise ValueError(
            f"route_distance_km must be between {MIN_DISTANCE_KM} and {MAX_DISTANCE_KM} km."
        )

    hours = parse_time_to_hours(time_of_day)
    if np.isnan(hours):
        raise ValueError(
            "time_of_day must be a valid 24-hour 'HH:MM' string or decimal hour in [0, 24)."
        )

    normalized_traffic = str(traffic_category).strip().title()
    if normalized_traffic not in TRAFFIC_CATEGORIES:
        raise ValueError(
            f"traffic_category must be one of {TRAFFIC_CATEGORIES}, got '{traffic_category}'."
        )

    if not (MIN_PREV_TIME_MIN <= float(previous_travel_time_min) <= MAX_PREV_TIME_MIN):
        raise ValueError(
            f"previous_travel_time_min must be between {MIN_PREV_TIME_MIN} and {MAX_PREV_TIME_MIN} min."
        )

    return pd.DataFrame(
        [
            {
                "route_distance_km": float(route_distance_km),
                "time_of_day": format_hours_to_hhmm(hours),
                "traffic_category": normalized_traffic,
                "previous_travel_time_min": float(previous_travel_time_min),
            }
        ]
    )
