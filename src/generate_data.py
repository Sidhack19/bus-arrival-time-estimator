"""Generate a realistic synthetic campus bus historical dataset.

The dataset mirrors real-world campus shuttle logs containing:
- route_distance_km: Remaining distance from bus to student's stop (km)
- time_of_day: 24-hour clock timestamp ('HH:MM')
- traffic_category: Observed campus road congestion ('Low', 'Medium', 'High')
- previous_travel_time_min: Travel time of the immediately preceding bus trip (min)
- arrival_time_min: Actual observed arrival duration (target, min)

A small controlled number of missing values and anomalous sensor rows are included
so data-cleaning steps in the ML pipeline can be demonstrated and verified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CAMPUS_ROUTES, DATA_DIR, RAW_DATA_PATH
from src.features import format_hours_to_hhmm


def _sample_traffic_for_hour(rng: np.random.Generator, hour: float) -> str:
    """Assign traffic category with realistic peak-hour probabilities."""
    if 8.0 <= hour <= 9.75 or 16.5 <= hour <= 18.25:
        probs = [0.12, 0.38, 0.50]  # Morning & evening rush
    elif 12.25 <= hour <= 14.0:
        probs = [0.20, 0.52, 0.28]  # Lunchtime campus movement
    elif 6.0 <= hour < 8.0 or 18.25 < hour <= 20.5:
        probs = [0.45, 0.42, 0.13]  # Shoulder hours
    else:
        probs = [0.78, 0.19, 0.03]  # Quiet early morning / late evening
    return str(rng.choice(["Low", "Medium", "High"], p=probs))


def _rush_delay_minutes(hour: float) -> float:
    """Smooth Gaussian peaks for morning, midday, and evening campus rush."""
    morning_peak = 3.2 * np.exp(-0.5 * ((hour - 8.85) / 0.65) ** 2)
    lunch_peak = 2.0 * np.exp(-0.5 * ((hour - 13.1) / 0.70) ** 2)
    evening_peak = 3.6 * np.exp(-0.5 * ((hour - 17.35) / 0.70) ** 2)
    return float(morning_peak + lunch_peak + evening_peak)


def generate_campus_bus_dataset(
    n_rows: int = 1200,
    seed: int = 42,
    output_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Create and save synthetic campus bus trip dataset to CSV."""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Traffic speed penalty factors
    traffic_multiplier = {"Low": 1.00, "Medium": 1.24, "High": 1.54}

    rows: list[dict[str, object]] = []
    # Mix active campus hours (06:30 to 22:00) with a slight peak bias
    for idx in range(n_rows):
        route = CAMPUS_ROUTES[idx % len(CAMPUS_ROUTES)]
        # Vary distance around campus stop segments (partial route to full route)
        fraction = rng.uniform(0.35, 1.20)
        distance_km = round(float(np.clip(route["distance_km"] * fraction, 0.6, 11.5)), 2)

        hour_decimal = float(rng.uniform(6.5, 22.0))
        time_str = format_hours_to_hhmm(hour_decimal)
        traffic = _sample_traffic_for_hour(rng, hour_decimal)

        # Base travel duration: ~25 km/h campus speed (2.4 min/km) + stop dwell (~1.6 min base)
        base_minutes = 1.6 + 2.35 * distance_km
        rush_delay = _rush_delay_minutes(hour_decimal)
        traffic_factor = traffic_multiplier[traffic]

        # Expected current travel duration before random trip-to-trip fluctuations
        expected_duration = base_minutes * traffic_factor + rush_delay

        # Previous bus trip experienced similar route conditions plus lagged variance
        prev_noise = rng.normal(0.0, 1.35)
        prev_time = round(float(np.clip(expected_duration * 0.96 + prev_noise, 2.0, 60.0)), 2)

        # Actual arrival time blends physical route factors + recent trip momentum + noise
        trip_noise = rng.normal(0.0, 1.05)
        actual_arrival = round(
            float(np.clip(0.62 * expected_duration + 0.38 * prev_time + trip_noise, 2.0, 65.0)),
            2,
        )

        rows.append(
            {
                "trip_id": f"TRIP-{idx + 1:04d}",
                "route_name": route["name"],
                "route_distance_km": distance_km,
                "time_of_day": time_str,
                "traffic_category": traffic,
                "previous_travel_time_min": prev_time,
                "arrival_time_min": actual_arrival,
            }
        )

    df = pd.DataFrame(rows)

    # Inject realistic beginner-dataset imperfections (approx 3% of rows)
    # 1. Missing previous_travel_time_min (e.g., first trip of the morning)
    missing_prev_idx = rng.choice(df.index, size=18, replace=False)
    df.loc[missing_prev_idx, "previous_travel_time_min"] = np.nan

    # 2. Missing traffic_category (sensor dropout)
    remaining_idx = df.index.difference(missing_prev_idx)
    missing_traffic_idx = rng.choice(remaining_idx, size=10, replace=False)
    df.loc[missing_traffic_idx, "traffic_category"] = np.nan

    # 3. Missing target arrival_time_min (unlogged trip)
    remaining_idx = remaining_idx.difference(missing_traffic_idx)
    missing_target_idx = rng.choice(remaining_idx, size=6, replace=False)
    df.loc[missing_target_idx, "arrival_time_min"] = np.nan

    # 4. Nonsensical/corrupted rows (negative distance, invalid time)
    remaining_idx = remaining_idx.difference(missing_target_idx)
    bad_rows = rng.choice(remaining_idx, size=6, replace=False)
    df.loc[bad_rows[0:2], "route_distance_km"] = -1.5
    df.loc[bad_rows[2:4], "time_of_day"] = "25:99"
    df.loc[bad_rows[4:6], "previous_travel_time_min"] = -4.0

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generated = generate_campus_bus_dataset()
    print(f"Generated {len(generated)} rows at {RAW_DATA_PATH}")
