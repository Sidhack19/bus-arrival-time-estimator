"""Automated tests for data quality, ML pipeline, CLI predictions, and web API."""

from __future__ import annotations

import numpy as np
import pytest

from app import create_app
from src.features import clean_raw_dataset, parse_time_to_hours, validate_prediction_inputs
from src.generate_data import generate_campus_bus_dataset
from src.predict import estimate_arrival_time, load_model_bundle
from src.train_model import train_and_evaluate


def test_parse_time_to_hours():
    assert parse_time_to_hours("08:30") == pytest.approx(8.5)
    assert parse_time_to_hours("17:45") == pytest.approx(17.75)
    assert parse_time_to_hours(12.25) == pytest.approx(12.25)
    assert np.isnan(parse_time_to_hours("25:99"))
    assert np.isnan(parse_time_to_hours(None))


def test_data_generation_and_cleaning(tmp_path):
    csv_path = tmp_path / "bus_trips.csv"
    raw_df = generate_campus_bus_dataset(n_rows=300, seed=7, output_path=csv_path)
    assert len(raw_df) == 300

    cleaned_df, report = clean_raw_dataset(raw_df)
    assert report.total_rows == 300
    assert report.missing_target_rows > 0
    assert report.invalid_distance_rows > 0
    assert report.invalid_time_rows > 0
    assert report.invalid_prev_time_rows > 0
    # Imputable missing feature values are kept for train-only imputation
    assert report.missing_feature_counts["previous_travel_time_min"] > 0
    assert report.missing_feature_counts["traffic_category"] > 0
    assert cleaned_df["arrival_time_min"].isna().sum() == 0


def test_end_to_end_training_and_prediction_sanity(tmp_path):
    csv_path = tmp_path / "bus.csv"
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    generate_campus_bus_dataset(n_rows=450, seed=42, output_path=csv_path)

    metadata = train_and_evaluate(
        data_path=csv_path,
        model_path=model_path,
        metadata_path=meta_path,
        random_state=42,
    )
    assert model_path.exists()
    assert meta_path.exists()
    assert metadata["test_metrics"]["r2"] > 0.85
    assert metadata["test_metrics"]["mae"] < 2.5

    load_model_bundle.cache_clear()
    low_traffic = estimate_arrival_time(
        route_distance_km=4.0,
        time_of_day="10:30",
        traffic_category="Low",
        previous_travel_time_min=11.0,
        model_path=model_path,
        metadata_path=meta_path,
    )
    high_traffic = estimate_arrival_time(
        route_distance_km=4.0,
        time_of_day="08:45",
        traffic_category="High",
        previous_travel_time_min=17.5,
        model_path=model_path,
        metadata_path=meta_path,
    )
    assert high_traffic.estimated_minutes > low_traffic.estimated_minutes
    assert low_traffic.lower_bound_minutes <= low_traffic.estimated_minutes <= low_traffic.upper_bound_minutes


def test_validate_prediction_inputs_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_prediction_inputs(-2.0, "08:30", "Low", 10.0)
    with pytest.raises(ValueError):
        validate_prediction_inputs(3.0, "29:10", "Low", 10.0)
    with pytest.raises(ValueError):
        validate_prediction_inputs(3.0, "08:30", "Extreme", 10.0)
    with pytest.raises(ValueError):
        validate_prediction_inputs(3.0, "08:30", "Low", 0.1)


def test_flask_api_endpoints(tmp_path):
    csv_path = tmp_path / "bus.csv"
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    generate_campus_bus_dataset(n_rows=350, seed=11, output_path=csv_path)
    train_and_evaluate(
        data_path=csv_path,
        model_path=model_path,
        metadata_path=meta_path,
        random_state=11,
    )
    load_model_bundle.cache_clear()

    flask_app = create_app(model_path=model_path, metadata_path=meta_path)
    client = flask_app.test_client()

    home_resp = client.get("/")
    assert home_resp.status_code == 200
    assert b"Campus Bus Arrival Time Estimator" in home_resp.data

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json()["model_ready"] is True

    pred_resp = client.post(
        "/api/predict",
        json={
            "route_distance_km": 3.5,
            "time_of_day": "08:45",
            "traffic_category": "High",
            "previous_travel_time_min": 14.0,
        },
    )
    assert pred_resp.status_code == 200
    body = pred_resp.get_json()
    assert body["estimated_minutes"] > 0
    assert body["traffic_category"] == "High"

    bad_resp = client.post(
        "/api/predict",
        json={
            "route_distance_km": 3.5,
            "time_of_day": "08:45",
            "traffic_category": "Gridlock",
            "previous_travel_time_min": 14.0,
        },
    )
    assert bad_resp.status_code == 400
