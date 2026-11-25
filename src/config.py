"""Shared configuration, schema definitions, and campus route presets."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "campus_bus_trips.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "bus_arrival_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAFFIC_CATEGORIES = ["Low", "Medium", "High"]
TARGET_COLUMN = "arrival_time_min"
MODEL_INPUT_COLUMNS = [
    "route_distance_km",
    "time_of_day",
    "traffic_category",
    "previous_travel_time_min",
]

# Sensible physical bounds for campus bus operations
MIN_DISTANCE_KM = 0.2
MAX_DISTANCE_KM = 30.0
MIN_PREV_TIME_MIN = 1.0
MAX_PREV_TIME_MIN = 120.0
MIN_ARRIVAL_TIME_MIN = 1.0
MAX_ARRIVAL_TIME_MIN = 120.0

# Campus route presets for the web UI and demo scenarios
CAMPUS_ROUTES = [
    {
        "id": "main-gate-hostels",
        "name": "Route A: Main Gate -> North Hostels",
        "distance_km": 3.2,
        "typical_prev_time_min": 11.5,
        "stops": 5,
    },
    {
        "id": "library-academic-loop",
        "name": "Route B: Central Library -> Academic Block Loop",
        "distance_km": 2.1,
        "typical_prev_time_min": 8.0,
        "stops": 4,
    },
    {
        "id": "station-campus-express",
        "name": "Route C: Railway Station -> Engineering Quad",
        "distance_km": 7.8,
        "typical_prev_time_min": 24.5,
        "stops": 7,
    },
    {
        "id": "sports-complex-shuttle",
        "name": "Route D: South Dorms -> Sports Complex",
        "distance_km": 4.9,
        "typical_prev_time_min": 16.0,
        "stops": 6,
    },
]
