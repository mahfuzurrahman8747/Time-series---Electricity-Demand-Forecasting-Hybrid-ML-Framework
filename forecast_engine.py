"""Model loading and recursive forecasting for the Streamlit app."""

import json
import math
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATHS = {
    "AutoARIMA (StatsForecast)": SAVED_MODELS_DIR / "M_SF.pkl",
    "Gradient Boosting": SAVED_MODELS_DIR / "M_GBoosting.pkl",
    "XGBoost": SAVED_MODELS_DIR / "M_XGBoost.pkl",
    "Residual ANN": SAVED_MODELS_DIR / "M_ANN.pth",
}
SCALER_PATH = SAVED_MODELS_DIR / "M_X_scaler.pkl"
FEATURES_PATH = SAVED_MODELS_DIR / "feature_name.json"
HOURLY_MEAN_PATH = SAVED_MODELS_DIR / "hourly_mean_demand.json"
MODEL_RESULTS_PATH = Path(__file__).resolve().parent / "model_results_top.csv"
RESULT_NAMES = {
    "AutoARIMA (StatsForecast)": "SARIMA",
    "Gradient Boosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "Residual ANN": "ANN",
}
MAX_FORECAST_HOURS = 24 * 365
MAX_FIXED_POINT_ITERATIONS = 200
PREDICTION_TOLERANCE_MW = 1.0


@st.cache_resource
def load_sarima_model():
    """Load the fitted StatsForecast model saved by final_model.ipynb."""
    model_path = MODEL_PATHS["AutoARIMA (StatsForecast)"]
    if not model_path.is_file():
        raise FileNotFoundError(f"SARIMA model file was not found: {model_path}")

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    expected_features = {"temp_mean", "wspd_mean", "lag1", "lag23"}
    model_features = set(getattr(model, "_exog", []))
    if model_features != expected_features:
        raise ValueError(
            "The saved AutoARIMA model uses an outdated or unstable feature set. "
            "Retrain it by running the final_model.ipynb training cells."
        )
    return model


@st.cache_data
def load_hourly_mean_demand():
    """Load the hourly demand defaults provided with the saved model artifacts."""
    if not HOURLY_MEAN_PATH.is_file():
        raise FileNotFoundError(
            f"Hourly mean demand file was not found: {HOURLY_MEAN_PATH}"
        )

    with HOURLY_MEAN_PATH.open(encoding="utf-8") as mean_file:
        hourly_means = json.load(mean_file)

    expected_hours = {str(hour) for hour in range(24)}
    if not isinstance(hourly_means, dict) or set(hourly_means) != expected_hours:
        raise ValueError(
            "Hourly mean demand JSON must contain numeric values for hours 0–23."
        )

    validated_means = {int(hour): float(value) for hour, value in hourly_means.items()}
    if not np.isfinite(list(validated_means.values())).all():
        raise ValueError("Hourly mean demand values must all be finite.")
    return validated_means


@st.cache_data
def load_feature_names():
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Model feature list was not found: {FEATURES_PATH}")

    with FEATURES_PATH.open(encoding="utf-8") as feature_file:
        feature_data = json.load(feature_file)
    features = feature_data.get("features")
    expected_features = {
        "hour_sin",
        "hour_cos",
        "temp_mean",
        "wspd_mean",
        "lag1",
        "lag23",
    }
    if not isinstance(features, list) or set(features) != expected_features:
        raise ValueError("The saved model feature list is missing expected inputs.")
    return features


@st.cache_resource
def load_forecast_model(model_name):
    """Load the selected fitted model and its feature preprocessing artifacts."""
    model_path = MODEL_PATHS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    if model_name == "AutoARIMA (StatsForecast)":
        return load_sarima_model(), None, None

    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Model feature scaler was not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    features = load_feature_names()
    model = (
        joblib.load(model_path)
        if model_name in {"Gradient Boosting", "XGBoost"}
        else None
    )
    if model_name == "Residual ANN":
        import torch
        from torch import nn

        class ResidualANN(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.input_layer = nn.Sequential(nn.Linear(input_size, 256), nn.GELU())
                self.block1 = nn.Sequential(
                    nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256)
                )
                self.block2 = nn.Sequential(
                    nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128)
                )
                self.block3 = nn.Sequential(
                    nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64)
                )
                self.output = nn.Sequential(
                    nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
                )
                self.proj1 = nn.Linear(256, 128)

            def forward(self, inputs):
                values = self.input_layer(inputs)
                residual = values
                values = nn.functional.gelu(self.block1(values) + residual)
                residual = self.proj1(values)
                values = nn.functional.gelu(self.block2(values) + residual)
                residual = values[:, :64]
                values = nn.functional.gelu(self.block3(values) + residual)
                return self.output(values)

        model = ResidualANN(input_size=len(features))
        state_dict = torch.load(
            MODEL_PATHS["Residual ANN"],
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ValueError(f"{model_name} expects an unexpected number of features.")
    return model, scaler, features


def next_forecast_time(model):
    """Return the hour immediately following the model's training data."""
    last_dates = getattr(model, "last_dates", None)
    if last_dates is None or len(last_dates) != 1:
        raise ValueError("Expected M_SF.pkl to contain one fitted time series.")

    return pd.Timestamp(last_dates[0]) + pd.Timedelta(hours=1)


def load_validation_r2(model_name):
    """Read the selected model's R² from its latest recorded test period."""
    if not MODEL_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Model evaluation results were not found: {MODEL_RESULTS_PATH}"
        )

    results = pd.read_csv(MODEL_RESULTS_PATH)
    if not {"Model", "R2", "Test_Period"}.issubset(results.columns):
        raise ValueError("Model results must include Model, R2, and Test_Period columns.")

    model_results = results.loc[
        results["Model"].astype("string").str.casefold()
        == RESULT_NAMES[model_name].casefold(),
        ["Test_Period", "R2"],
    ]
    if model_results.empty:
        raise ValueError(f"No test-set R² value was found for {model_name}.")

    period_ends = pd.to_datetime(
        model_results["Test_Period"]
        .astype("string")
        .str.rsplit(" to ", n=1)
        .str[-1],
        errors="coerce",
    )
    if period_ends.isna().any():
        raise ValueError(f"Invalid test period recorded for {model_name}.")

    latest_results = model_results.loc[period_ends == period_ends.max()]
    latest_r2 = pd.to_numeric(latest_results["R2"], errors="coerce")
    if len(latest_r2) != 1 or not np.isfinite(latest_r2.iloc[0]):
        raise ValueError(f"Expected one finite test-set R² value for {model_name}.")

    return float(latest_r2.iloc[0])


def predict_ml_demand_at(
    model_name,
    model,
    scaler,
    features,
    history,
    forecast_times,
    temperature,
    wind_speed,
    last_hour_demand,
    progress,
):
    """Forecast each hour in order for a tree or neural-network model."""
    demand_history = list(np.asarray(history[:, 0], dtype=float))
    if len(demand_history) < 23:
        raise ValueError("The saved model has insufficient history for lag23.")

    prediction = float("nan")
    progress_interval = max(len(forecast_times) // 100, 1)
    for index, forecast_time in enumerate(forecast_times):
        hour = forecast_time.hour
        feature_values = {
            "hour_sin": math.sin(2 * math.pi * hour / 23),
            "hour_cos": math.cos(2 * math.pi * hour / 23),
            "temp_mean": float(temperature),
            "wspd_mean": float(wind_speed),
            "lag1": (
                float(last_hour_demand)
                if index == len(forecast_times) - 1
                else demand_history[-1]
            ),
            "lag23": demand_history[-23],
        }
        model_input = pd.DataFrame(
            [[feature_values[feature] for feature in features]],
            columns=features,
        )
        scaled_input = scaler.transform(model_input)
        if model_name == "Residual ANN":
            import torch

            with torch.inference_mode():
                prediction = float(
                    model(torch.tensor(scaled_input, dtype=torch.float32))
                    .reshape(-1)[0]
                    .item()
                )
        else:
            prediction = float(model.predict(scaled_input)[0])

        if not math.isfinite(prediction):
            raise ValueError(f"{model_name} returned a non-finite forecast.")
        demand_history.append(prediction)
        completed = index + 1
        if completed % progress_interval == 0 or completed == len(forecast_times):
            progress.progress(
                completed / len(forecast_times),
                text=(
                    f"Forecasting with {model_name} "
                    f"({completed:,}/{len(forecast_times):,} hours)"
                ),
            )

    return prediction, len(forecast_times)


def build_future_exog(
    model,
    forecast_times,
    predicted_demand,
    temperature,
    wind_speed,
):
    """Build hourly exogenous rows, recursively deriving demand-lag features."""
    required_features = {
        "temp_mean",
        "wspd_mean",
        "lag1",
        "lag23",
    }
    model_features = getattr(model, "_exog", None)
    if model_features is None or not required_features.issubset(model_features):
        raise ValueError(
            "The saved model does not expose the expected SARIMA regressors: "
            f"{sorted(required_features)}."
        )

    history = getattr(getattr(model, "ga", None), "data", None)
    if history is None or len(history) < 23:
        raise ValueError("The saved model has insufficient demand history for lag23.")

    forecast_times = pd.DatetimeIndex(forecast_times)
    predicted_demand = np.asarray(predicted_demand, dtype=float)
    if len(forecast_times) != len(predicted_demand):
        raise ValueError("Each forecast hour must have a corresponding demand estimate.")

    historical_demand = np.asarray(history[:, 0], dtype=float)
    combined_demand = np.concatenate((historical_demand, predicted_demand))
    row_indices = len(historical_demand) + np.arange(len(forecast_times))
    values = {
        "temp_mean": np.full(len(forecast_times), float(temperature)),
        "wspd_mean": np.full(len(forecast_times), float(wind_speed)),
        "lag1": combined_demand[row_indices - 1],
        "lag23": combined_demand[row_indices - 23],
    }

    return pd.DataFrame(
        {
            "unique_id": model.uids[0],
            "ds": forecast_times,
            **{feature: values[feature] for feature in model_features},
        }
    )


def predict_demand_at(
    model,
    forecast_time,
    temperature,
    wind_speed,
    last_hour_demand,
    progress,
):
    """Forecast to a selected hour, feeding predicted demand back into its lags."""
    first_forecast_time = next_forecast_time(model)
    if forecast_time < first_forecast_time:
        raise ValueError(
            "Choose a forecast time after the model's final training timestamp "
            f"({pd.Timestamp(model.last_dates[0]):%Y-%m-%d %H:%M})."
        )

    horizon = int((forecast_time - first_forecast_time) / pd.Timedelta(hours=1)) + 1
    if horizon > MAX_FORECAST_HOURS:
        raise ValueError(
            "The selected time is more than one year beyond the model's training "
            "data. Select a closer time or retrain with newer data."
        )

    forecast_times = pd.date_range(first_forecast_time, periods=horizon, freq="h")
    historical_demand = np.asarray(model.ga.data[:, 0], dtype=float)
    predicted_demand = np.full(horizon, historical_demand[-1], dtype=float)
    forecast_name = model.models[0].alias

    for iteration in range(1, MAX_FIXED_POINT_ITERATIONS + 1):
        future_exog = build_future_exog(
            model,
            forecast_times,
            predicted_demand,
            temperature,
            wind_speed,
        )
        future_exog.loc[future_exog.index[-1], "lag1"] = float(last_hour_demand)
        forecasts = model.predict(h=horizon, X_df=future_exog, level=[80, 90])
        updated_predictions = forecasts[forecast_name].to_numpy(dtype=float)
        if not np.isfinite(updated_predictions).all():
            raise ValueError("The model returned a non-finite demand forecast.")

        difference = float(np.max(np.abs(updated_predictions - predicted_demand)))
        progress.progress(
            iteration / MAX_FIXED_POINT_ITERATIONS,
            text=(
                f"Refining recursive forecast ({iteration}/"
                f"{MAX_FIXED_POINT_ITERATIONS}; {horizon:,} hourly steps)"
            ),
        )
        predicted_demand = updated_predictions
        if difference <= PREDICTION_TOLERANCE_MW:
            return forecasts.iloc[-1], horizon, future_exog.iloc[-1]

    raise RuntimeError(
        "The recursive forecast did not converge. Select a closer forecast time "
        "or retrain the model with newer data."
    )
