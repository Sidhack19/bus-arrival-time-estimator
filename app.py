"""Flask web application for the Bus Arrival Time Estimator."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.config import CAMPUS_ROUTES, METADATA_PATH, MODEL_PATH, TRAFFIC_CATEGORIES
from src.predict import estimate_arrival_time, load_model_bundle
from src.train_model import train_and_evaluate

BASE_DIR = Path(__file__).resolve().parent


def create_app(
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["MODEL_PATH"] = model_path
    app.config["METADATA_PATH"] = metadata_path

    if not model_path.exists():
        train_and_evaluate(model_path=model_path, metadata_path=metadata_path)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            routes=CAMPUS_ROUTES,
            traffic_categories=TRAFFIC_CATEGORIES,
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_ready": Path(app.config["MODEL_PATH"]).exists()})

    @app.get("/api/routes")
    def get_routes():
        return jsonify({"routes": CAMPUS_ROUTES, "traffic_categories": TRAFFIC_CATEGORIES})

    @app.get("/api/metadata")
    def get_metadata():
        meta_file = Path(app.config["METADATA_PATH"])
        if meta_file.exists():
            return jsonify(json.loads(meta_file.read_text(encoding="utf-8")))
        _, meta = load_model_bundle(
            str(app.config["MODEL_PATH"]), str(app.config["METADATA_PATH"])
        )
        return jsonify(meta)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        try:
            distance_km = float(payload["route_distance_km"])
            time_of_day = payload["time_of_day"]
            traffic_category = str(payload["traffic_category"])
            prev_time_min = float(payload["previous_travel_time_min"])
        except (KeyError, TypeError, ValueError) as exc:
            return (
                jsonify(
                    {
                        "error": "Please provide valid route_distance_km, time_of_day, traffic_category, and previous_travel_time_min.",
                        "details": str(exc),
                    }
                ),
                400,
            )

        try:
            result = estimate_arrival_time(
                route_distance_km=distance_km,
                time_of_day=time_of_day,
                traffic_category=traffic_category,
                previous_travel_time_min=prev_time_min,
                model_path=Path(app.config["MODEL_PATH"]),
                metadata_path=Path(app.config["METADATA_PATH"]),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(result.to_dict())

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="127.0.0.1", port=5000, debug=False)
