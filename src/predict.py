"""Inference and beginner-friendly CLI predictor for campus bus arrival times."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.config import METADATA_PATH, MIN_ARRIVAL_TIME_MIN, MODEL_PATH, TRAFFIC_CATEGORIES
from src.features import (
    format_hours_to_hhmm,
    is_campus_rush_hour,
    parse_time_to_hours,
    validate_prediction_inputs,
)


@dataclass
class PredictionResult:
    """Structured prediction output for CLI and Web UI."""

    estimated_minutes: float
    lower_bound_minutes: float
    upper_bound_minutes: float
    uncertainty_margin_minutes: float
    departure_time: str
    estimated_arrival_clock: str
    is_peak_hour: bool
    traffic_category: str
    route_distance_km: float
    previous_travel_time_min: float
    model_name: str
    explanation: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_minutes": self.estimated_minutes,
            "lower_bound_minutes": self.lower_bound_minutes,
            "upper_bound_minutes": self.upper_bound_minutes,
            "uncertainty_margin_minutes": self.uncertainty_margin_minutes,
            "departure_time": self.departure_time,
            "estimated_arrival_clock": self.estimated_arrival_clock,
            "is_peak_hour": self.is_peak_hour,
            "traffic_category": self.traffic_category,
            "route_distance_km": self.route_distance_km,
            "previous_travel_time_min": self.previous_travel_time_min,
            "model_name": self.model_name,
            "explanation": self.explanation,
        }


@lru_cache(maxsize=2)
def load_model_bundle(
    model_path: str = str(MODEL_PATH), metadata_path: str = str(METADATA_PATH)
) -> tuple[Any, dict[str, Any]]:
    """Load the persisted sklearn Pipeline and metadata JSON."""
    m_path = Path(model_path)
    meta_path = Path(metadata_path)
    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {m_path}. Run `python -m src.train_model` first."
        )
    model = joblib.load(m_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, metadata


def _build_explanation(
    distance_km: float,
    hours: float,
    traffic: str,
    prev_time: float,
    eta: float,
) -> list[str]:
    """Generate clear beginner-friendly notes explaining the estimate."""
    reasons: list[str] = [
        f"Route distance of {distance_km:.1f} km sets the base campus travel duration."
    ]
    if is_campus_rush_hour(hours):
        reasons.append(
            "Current time falls in a campus peak window (class change/rush), adding boarding and road delay."
        )
    else:
        reasons.append("Current time is outside campus rush hours, keeping dwell delays lower.")

    if traffic == "High":
        reasons.append("High traffic increases expected intersection and shuttle delays.")
    elif traffic == "Medium":
        reasons.append("Medium traffic adds moderate slowdowns along campus stops.")
    else:
        reasons.append("Low traffic allows smooth travel near free-flow campus speed.")

    diff = eta - prev_time
    if abs(diff) < 1.0:
        reasons.append(
            f"Prediction closely tracks the previous bus travel time ({prev_time:.1f} min)."
        )
    elif diff > 0:
        reasons.append(
            f"Estimate is {diff:.1f} min slower than the previous trip ({prev_time:.1f} min) due to current conditions."
        )
    else:
        reasons.append(
            f"Estimate is {abs(diff):.1f} min faster than the previous trip ({prev_time:.1f} min)."
        )
    return reasons


def estimate_arrival_time(
    route_distance_km: float,
    time_of_day: str | float,
    traffic_category: str,
    previous_travel_time_min: float,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> PredictionResult:
    """Predict campus bus arrival time in minutes and expected clock arrival."""
    input_df = validate_prediction_inputs(
        route_distance_km=route_distance_km,
        time_of_day=time_of_day,
        traffic_category=traffic_category,
        previous_travel_time_min=previous_travel_time_min,
    )
    model, metadata = load_model_bundle(str(model_path), str(metadata_path))

    raw_pred = float(model.predict(input_df)[0])
    eta = round(max(MIN_ARRIVAL_TIME_MIN, raw_pred), 1)

    mae = float(metadata.get("test_metrics", {}).get("mae", 1.5))
    margin = round(max(0.8, mae), 1)
    lower = round(max(MIN_ARRIVAL_TIME_MIN, eta - margin), 1)
    upper = round(eta + margin, 1)

    hours = parse_time_to_hours(input_df.iloc[0]["time_of_day"])
    arrival_hours = hours + eta / 60.0
    normalized_traffic = str(input_df.iloc[0]["traffic_category"])

    return PredictionResult(
        estimated_minutes=eta,
        lower_bound_minutes=lower,
        upper_bound_minutes=upper,
        uncertainty_margin_minutes=margin,
        departure_time=str(input_df.iloc[0]["time_of_day"]),
        estimated_arrival_clock=format_hours_to_hhmm(arrival_hours),
        is_peak_hour=is_campus_rush_hour(hours),
        traffic_category=normalized_traffic,
        route_distance_km=float(route_distance_km),
        previous_travel_time_min=float(previous_travel_time_min),
        model_name=str(metadata.get("best_model", "Regression Model")),
        explanation=_build_explanation(
            float(route_distance_km),
            hours,
            normalized_traffic,
            float(previous_travel_time_min),
            eta,
        ),
    )


def _prompt_interactive() -> tuple[float, str, str, float]:
    """Collect validated inputs interactively from a terminal user."""
    print("\n=== Campus Bus Arrival Time Estimator ===")
    distance = float(input("1. Route distance in km (e.g. 3.5): ").strip())
    time_str = input("2. Time of day in 24h HH:MM (e.g. 08:45): ").strip()
    traffic = input(f"3. Traffic category {TRAFFIC_CATEGORIES} (e.g. Medium): ").strip()
    prev_time = float(input("4. Previous trip travel time in minutes (e.g. 12.5): ").strip())
    return distance, time_str, traffic, prev_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate campus bus arrival time.")
    parser.add_argument("--distance", type=float, help="Route distance in km (e.g. 3.2)")
    parser.add_argument("--time", type=str, help="Time of day in HH:MM 24h format (e.g. 08:45)")
    parser.add_argument(
        "--traffic",
        type=str,
        choices=TRAFFIC_CATEGORIES,
        help="Traffic category: Low, Medium, or High",
    )
    parser.add_argument(
        "--prev-time", type=float, help="Previous bus travel time in minutes (e.g. 12.0)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for trip details interactively",
    )
    args = parser.parse_args()

    if args.interactive or any(
        val is None for val in (args.distance, args.time, args.traffic, args.prev_time)
    ):
        distance, time_str, traffic, prev_time = _prompt_interactive()
    else:
        distance, time_str, traffic, prev_time = (
            args.distance,
            args.time,
            args.traffic,
            args.prev_time,
        )

    result = estimate_arrival_time(distance, time_str, traffic, prev_time)
    print("\n--- Prediction Summary ---")
    print(f"Model Used           : {result.model_name}")
    print(f"Estimated Wait Time  : {result.estimated_minutes:.1f} minutes")
    print(f"Expected Range (±MAE): {result.lower_bound_minutes:.1f} - {result.upper_bound_minutes:.1f} min")
    print(f"Est. Arrival Clock   : {result.estimated_arrival_clock}")
    print("Key Factors:")
    for item in result.explanation:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
