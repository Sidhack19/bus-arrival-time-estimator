"""Train, compare, evaluate, and persist campus bus arrival regression models."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.config import (
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_INPUT_COLUMNS,
    MODEL_PATH,
    MODELS_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TRAFFIC_CATEGORIES,
)
from src.features import (
    build_preprocessor,
    clean_raw_dataset,
    get_transformed_feature_names,
    is_campus_rush_hour,
    parse_time_to_hours,
)
from src.generate_data import generate_campus_bus_dataset

if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Return MAE, RMSE, and R2 score for regression predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def bootstrap_confidence_intervals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Estimate 95% bootstrap confidence intervals for test MAE and RMSE."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)

    maes = np.empty(n_resamples, dtype=float)
    rmses = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        err = y_true_arr[idx] - y_pred_arr[idx]
        maes[i] = np.mean(np.abs(err))
        rmses[i] = np.sqrt(np.mean(err**2))

    return {
        "mae_95ci": [round(float(np.percentile(maes, 2.5)), 4), round(float(np.percentile(maes, 97.5)), 4)],
        "rmse_95ci": [round(float(np.percentile(rmses, 2.5)), 4), round(float(np.percentile(rmses, 97.5)), 4)],
    }


def build_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Define baseline and regression candidate pipelines."""
    estimators: dict[str, Any] = {
        "Naive Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Polynomial Ridge (deg=2)": Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=8.0)),
            ]
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=4,
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }


def evaluate_slice_performance(
    X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Break down test errors by traffic category and rush-hour status."""
    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred
    eval_df["is_rush"] = eval_df["time_of_day"].apply(
        lambda t: "Peak Hour" if is_campus_rush_hour(parse_time_to_hours(t)) else "Off-Peak"
    )

    slices: dict[str, dict[str, Any]] = {"by_traffic": {}, "by_time_window": {}}
    for cat in TRAFFIC_CATEGORIES:
        sub = eval_df[eval_df["traffic_category"] == cat]
        if len(sub) > 0:
            metrics = compute_regression_metrics(sub["y_true"], sub["y_pred"])
            metrics["count"] = int(len(sub))
            slices["by_traffic"][cat] = metrics

    for window in ["Peak Hour", "Off-Peak"]:
        sub = eval_df[eval_df["is_rush"] == window]
        if len(sub) > 0:
            metrics = compute_regression_metrics(sub["y_true"], sub["y_pred"])
            metrics["count"] = int(len(sub))
            slices["by_time_window"][window] = metrics

    return slices


def extract_feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    """Extract normalized interpretability scores from linear or tree models."""
    reg = pipeline.named_steps["regressor"]
    names = get_transformed_feature_names()

    if hasattr(reg, "coef_"):
        raw = np.abs(np.asarray(reg.coef_, dtype=float).ravel())[: len(names)]
    elif hasattr(reg, "feature_importances_"):
        raw = np.asarray(reg.feature_importances_, dtype=float)
    elif isinstance(reg, Pipeline) and "ridge" in reg.named_steps:
        poly = reg.named_steps["poly"]
        poly_names = poly.get_feature_names_out(names)
        coefs = np.abs(reg.named_steps["ridge"].coef_)
        top_idx = np.argsort(coefs)[::-1][:7]
        total = float(coefs[top_idx].sum()) or 1.0
        return [
            {
                "feature": str(poly_names[i]),
                "importance": round(float(coefs[i] / total), 4),
            }
            for i in top_idx
        ]
    else:
        return []

    total = float(raw.sum()) or 1.0
    pairs = sorted(
        zip(names, raw / total), key=lambda item: item[1], reverse=True
    )
    return [
        {"feature": name, "importance": round(float(score), 4)}
        for name, score in pairs
    ]


def save_diagnostic_plots(
    cleaned_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    best_model_name: str,
    figures_dir: Path = FIGURES_DIR,
) -> list[str]:
    """Generate and save EDA and model evaluation plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_paths: list[str] = []

    # 1. Target distribution (histogram + boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(cleaned_df[TARGET_COLUMN], bins=24, kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title("Arrival Time Distribution")
    axes[0].set_xlabel("Arrival Time (minutes)")
    sns.boxplot(x=cleaned_df[TARGET_COLUMN], ax=axes[1], color="#60a5fa")
    axes[1].set_title("Arrival Time Spread & Outliers")
    axes[1].set_xlabel("Arrival Time (minutes)")
    plt.tight_layout()
    p1 = figures_dir / "target_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p1))

    # 2. Relationships: distance & time of day vs arrival time
    plot_df = cleaned_df.copy()
    plot_df["hour_decimal"] = plot_df["time_of_day"].apply(parse_time_to_hours)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(
        data=plot_df,
        x="route_distance_km",
        y=TARGET_COLUMN,
        hue="traffic_category",
        hue_order=TRAFFIC_CATEGORIES,
        alpha=0.72,
        ax=axes[0],
    )
    axes[0].set_title("Distance vs Arrival Time by Traffic")
    axes[0].set_xlabel("Route Distance (km)")
    axes[0].set_ylabel("Arrival Time (minutes)")

    sns.scatterplot(
        data=plot_df,
        x="hour_decimal",
        y=TARGET_COLUMN,
        hue="traffic_category",
        hue_order=TRAFFIC_CATEGORIES,
        alpha=0.65,
        ax=axes[1],
    )
    axes[1].set_title("Time of Day vs Arrival Time")
    axes[1].set_xlabel("Hour of Day (24h)")
    axes[1].set_ylabel("Arrival Time (minutes)")
    plt.tight_layout()
    p2 = figures_dir / "eda_relationships.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p2))

    # 3. Model Comparison (Validation MAE & R2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ordered = comparison_df.sort_values("val_mae")
    sns.barplot(data=ordered, x="val_mae", y="model", ax=axes[0], palette="Blues_r", hue="model", legend=False)
    axes[0].set_title("Validation MAE (Lower is Better)")
    axes[0].set_xlabel("Mean Absolute Error (minutes)")
    axes[0].set_ylabel("")

    sns.barplot(data=ordered, x="val_r2", y="model", ax=axes[1], palette="Greens", hue="model", legend=False)
    axes[1].set_title("Validation R² Score (Higher is Better)")
    axes[1].set_xlabel("R² Score")
    axes[1].set_ylabel("")
    plt.tight_layout()
    p3 = figures_dir / "model_comparison.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p3))

    # 4. Actual vs Predicted & Residuals on held-out Test Set
    residuals = y_test.values - y_test_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_test.values, y_test_pred, alpha=0.68, color="#2563eb", edgecolors="white", linewidth=0.5)
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    axes[0].plot(lims, lims, "--", color="#dc2626", linewidth=1.6, label="Ideal Prediction")
    axes[0].set_title(f"{best_model_name}: Actual vs Predicted (Test Set)")
    axes[0].set_xlabel("Actual Arrival Time (minutes)")
    axes[0].set_ylabel("Predicted Arrival Time (minutes)")
    axes[0].legend()

    sns.histplot(residuals, bins=22, kde=True, ax=axes[1], color="#0d9488")
    axes[1].axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.4)
    axes[1].set_title("Test Residual Errors (Actual - Predicted)")
    axes[1].set_xlabel("Residual Error (minutes)")
    plt.tight_layout()
    p4 = figures_dir / "actual_vs_predicted_and_residuals.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p4))

    return saved_paths


def train_and_evaluate(
    data_path: Path = RAW_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run complete data loading, cleaning, split, model selection, and artifact saving."""
    if not data_path.exists():
        generate_campus_bus_dataset(output_path=data_path, seed=random_state)

    raw_df = pd.read_csv(data_path)
    cleaned_df, quality_report = clean_raw_dataset(raw_df)

    X = cleaned_df[MODEL_INPUT_COLUMNS].copy()
    y = cleaned_df[TARGET_COLUMN].copy()

    # 70% Train / 15% Validation / 15% Test split BEFORE fitting any preprocessing
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    val_fraction_of_remaining = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction_of_remaining, random_state=random_state
    )

    candidates = build_candidate_models(random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in candidates.items():
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=("neg_mean_absolute_error", "r2"),
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        train_metrics = compute_regression_metrics(y_train, pipeline.predict(X_train))
        val_metrics = compute_regression_metrics(y_val, pipeline.predict(X_val))

        comparison_rows.append(
            {
                "model": name,
                "cv_mae": round(float(-cv_results["test_neg_mean_absolute_error"].mean()), 4),
                "cv_r2": round(float(cv_results["test_r2"].mean()), 4),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("val_mae").reset_index(drop=True)
    # Exclude naive baseline when choosing best ML model
    non_baseline = comparison_df[comparison_df["model"] != "Naive Mean Baseline"]
    best_model_name = str(non_baseline.iloc[0]["model"])

    # Retrain selected model on combined Train + Validation before final held-out Test check
    final_pipeline = build_candidate_models(random_state=random_state)[best_model_name]
    final_pipeline.fit(X_train_val, y_train_val)

    y_test_pred = final_pipeline.predict(X_test)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)
    test_ci = bootstrap_confidence_intervals(y_test, y_test_pred)
    slice_metrics = evaluate_slice_performance(X_test, y_test, y_test_pred)
    feature_importance = extract_feature_importance(final_pipeline)

    plot_paths = save_diagnostic_plots(
        cleaned_df=cleaned_df,
        comparison_df=comparison_df,
        y_test=y_test,
        y_test_pred=y_test_pred,
        best_model_name=best_model_name,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    metadata = {
        "best_model": best_model_name,
        "features": MODEL_INPUT_COLUMNS,
        "target": TARGET_COLUMN,
        "dataset_splits": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_plus_val_rows": int(len(X_train_val)),
        },
        "data_quality": asdict(quality_report),
        "model_comparison": comparison_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "test_confidence_intervals": test_ci,
        "slice_metrics": slice_metrics,
        "feature_importance": feature_importance,
        "diagnostic_plots": plot_paths,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"Best model: {summary['best_model']}")
    print("Model comparison (sorted by validation MAE):")
    print(pd.DataFrame(summary["model_comparison"]).to_string(index=False))
    print(f"\nHeld-out Test Metrics: {summary['test_metrics']}")
    print(f"95% Bootstrap CIs: {summary['test_confidence_intervals']}")
    print(f"Saved model to: {MODEL_PATH}")
