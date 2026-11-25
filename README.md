# Bus Arrival Time Estimator

A beginner-friendly machine learning prediction system that estimates campus bus arrival times so students do not have to wait unnecessarily at bus stops.

## Problem Statement
Students may wait unnecessarily because they do not know the approximate arrival time of a campus bus.

## Methodology
The system uses four core trip features to train and compare regression models:
1. **Route Distance (`route_distance_km`)**: Remaining distance between the bus and the stop (km).
2. **Time of Day (`time_of_day`)**: 24-hour clock timestamp (`HH:MM`), transformed into decimal hours, cyclical sine/cosine encodings, and a campus rush-hour indicator.
3. **Traffic Category (`traffic_category`)**: Ordinal congestion level (`Low`, `Medium`, `High`).
4. **Previous Travel Time (`previous_travel_time_min`)**: Observed duration of the preceding trip on the same route (minutes).

### Machine Learning Pipeline
- **Data Quality Audit**: Identifies and drops corrupted/impossible rows and targetless rows, while preserving imputable missing feature values.
- **Leakage-Free Splitting**: Splits data into **70% Train / 15% Validation / 15% Test** *before* fitting any imputer, encoder, or scaler.
- **Models Evaluated**:
  - Naive Mean Baseline (`DummyRegressor`)
  - Linear Regression
  - Ridge Regression
  - Polynomial Ridge (`degree=2`)
  - Random Forest Regressor
- **Evaluation**: Compares 5-fold cross-validation and validation MAE / RMSE / R², then evaluates the winning model once on the held-out test set with 95% bootstrap confidence intervals and slice-level breakdowns by traffic category and peak hours.

## Project Structure

```text
bus-arrival-time-estimator/
├── app.py                              # Flask web UI & JSON prediction API
├── requirements.txt                    # Python dependencies
├── data/
│   └── campus_bus_trips.csv            # Generated synthetic campus shuttle dataset
├── models/
│   ├── bus_arrival_model.joblib        # Saved best sklearn Pipeline
│   └── model_metadata.json             # Metrics, CIs, slices, and feature importances
├── notebooks/
│   └── bus_arrival_estimator.ipynb     # Guided EDA + ML story notebook
├── reports/figures/                    # Saved EDA & evaluation plots
├── src/
│   ├── config.py                       # Schema, paths, bounds, and route presets
│   ├── features.py                     # Time parsing, data cleaning, sklearn preprocessor
│   ├── generate_data.py                # Reproducible synthetic dataset generator
│   ├── train_model.py                  # Training, model comparison, plots, persistence
│   └── predict.py                      # Prediction helper & interactive CLI
├── static/                             # Frontend CSS & JS
├── templates/index.html                # Web dashboard template
└── tests/test_estimator.py             # Automated pytest suite
```

## Quick Start

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Generate dataset and train/compare regression models
python -m src.generate_data
python -m src.train_model

# 3. Predict via CLI (arguments or interactive)
python -m src.predict --distance 3.5 --time 08:45 --traffic High --prev-time 14.0
python -m src.predict --interactive

# 4. Launch the web prediction dashboard
python app.py
# Open http://127.0.0.1:5000 in your browser

# 5. Run automated tests
pytest -v
```
