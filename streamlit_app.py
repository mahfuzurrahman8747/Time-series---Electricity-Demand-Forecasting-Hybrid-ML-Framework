"""
⚡ Electricity Demand Forecasting — Streamlit App
Run with: streamlit run streamlit_app.py
"""

import math
import pickle
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ================================================================================
# FUNCTIONS
# ================================================================================

@st.cache_resource
def load_gb_model():
    """Load the trained Gradient Boosting model, scaler, and feature list."""
    models_dir = Path(__file__).resolve().parent / "saved_models"

    model_path = models_dir / "M_GBoosting.pkl"
    scaler_path = models_dir / "M_X_scaler.pkl"

    feature_candidates = [
        models_dir / "feature_name.json",
        models_dir / "feature_names.json",
        models_dir / "model_metadata.json",
    ]

    feature_names = None
    for feature_file in feature_candidates:
        if not feature_file.exists():
            continue
        with open(feature_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            if "features" in payload:
                feature_names = payload["features"]
            elif "feature_names" in payload:
                feature_names = payload["feature_names"]
            elif "model_name" in payload and "features" in payload:
                feature_names = payload["features"]

        if isinstance(payload, list):
            feature_names = payload

        if feature_names is not None:
            break

    if feature_names is None:
        raise FileNotFoundError(
            "Feature metadata was not found in saved_models. "
            "Expected a feature_name.json or feature_names.json file."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, feature_names


@st.cache_resource
def load_hourly_mean_demand():
    """Load average hourly demand by hour of day."""
    models_dir = Path(__file__).resolve().parent / "saved_models"
    hourly_means_path = models_dir / "hourly_mean_demand.json"

    if hourly_means_path.exists():
        with open(hourly_means_path, "r", encoding="utf-8") as f:
            hourly_means = json.load(f)
        return {int(k): float(v) for k, v in hourly_means.items()}

    # Fallback if the historical file is missing.
    return {hour: 9500.0 for hour in range(24)}


def compute_cyclic(value: float, period: float):
    """Return sine/cosine encoding for a cyclic feature."""
    return math.sin(2 * math.pi * value / period), math.cos(2 * math.pi * value / period)


def build_feature_row(selected_date, hour, temperature, wind_speed, demand_lag1):
    """Create a single-row feature dictionary matching the trained model schema."""
    day_of_week = selected_date.weekday()
    month = selected_date.month
    day_of_year = selected_date.timetuple().tm_yday

    hour_sin, hour_cos = compute_cyclic(hour, 24)
    dow_sin, dow_cos = compute_cyclic(day_of_week, 7)
    month_sin, month_cos = compute_cyclic(month, 12)
    doy_sin, doy_cos = compute_cyclic(day_of_year, 365)

    rolling_std = max(5.0, abs(float(demand_lag1)) * 0.05)
    base_demand = float(demand_lag1)

    feature_row = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "load_shedding": 0.0,
        "temp_mean": float(temperature),
        "wspd_mean": float(wind_speed),
        "lag1": base_demand,
        "lag2": base_demand,
        "lag3": base_demand,
        "lag6": base_demand,
        "lag12": base_demand,
        "lag24": base_demand,
        "lag48": base_demand,
        "lag72": base_demand,
        "lag168": base_demand,
        "lag336": base_demand,
        "roll6_mean": base_demand,
        "roll6_std": rolling_std,
        "roll12_mean": base_demand,
        "roll12_std": rolling_std,
        "roll24_mean": base_demand,
        "roll24_std": rolling_std,
        "roll48_mean": base_demand,
        "roll48_std": rolling_std,
        "roll168_mean": base_demand,
        "roll168_std": rolling_std,
    }
    return feature_row


def predict_demand(model, scaler, features, input_dict):
    """Generate a model prediction from a feature dictionary."""
    try:
        input_df = pd.DataFrame([input_dict], columns=features)
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        return float(prediction)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# ================================================================================
# PAGE CONFIG
# ================================================================================

st.set_page_config(
    page_title="⚡ Electricity Demand Forecasting",
    page_icon="⚡",
    layout="centered",
)

# ================================================================================
# LOAD MODEL & HOURLY MEANS
# ================================================================================

try:
    model, scaler, features = load_gb_model()
    hourly_means = load_hourly_mean_demand()
    st.sidebar.success("✅ Model & Hourly Means Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ================================================================================
# SIDEBAR - INPUTS
# ================================================================================

st.sidebar.title("⚙️ Input Parameters")

selected_date = st.sidebar.date_input("📅 Select Date", value=datetime.now())
hour = st.sidebar.slider("⏰ Hour (0-23)", min_value=0, max_value=23, value=12)
temperature = st.sidebar.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
wind_speed = st.sidebar.number_input("💨 Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

st.sidebar.subheader("📊 Historical Demand (MW)")
default_demand = float(hourly_means.get(hour, 9500.0))
st.sidebar.info(f"📈 Hour {hour:02d}:00 — Last Year Average: **{default_demand:.1f} MW**")

use_default = st.sidebar.checkbox("I don't have last hour demand — use historical average", value=False)

if use_default:
    demand_lag1 = default_demand
    st.sidebar.success(f"✅ Using historical average: **{default_demand:.1f} MW**")
else:
    demand_lag1 = st.sidebar.number_input(
        "Last Hour Demand (MW)",
        min_value=0.0,
        max_value=20000.0,
        value=default_demand,
        step=10.0,
        help="Leave as default or enter actual demand value",
    )

# ================================================================================
# MAIN DASHBOARD
# ================================================================================

st.title("⚡ Electricity Demand Forecasting")
st.divider()
st.metric("📊 Model Accuracy", "98.255%")
st.divider()

st.subheader("🔮 Make Prediction")

if st.button("🚀 Predict Demand", use_container_width=True):
    input_data = build_feature_row(selected_date, hour, temperature, wind_speed, demand_lag1)
    prediction = predict_demand(model, scaler, features, input_data)

    if prediction is not None:
        st.success(f"✅ **Predicted Demand: {prediction:,.1f} MW**")

        day_name = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ][selected_date.weekday()]

        st.info(
            f"""
            **Input Summary:**
            - Date: {selected_date.strftime('%Y-%m-%d')} ({day_name})
            - Hour: {hour:02d}:00
            - Temperature: {temperature}°C
            - Wind Speed: {wind_speed} m/s
            - Last Hour Demand: {demand_lag1:,.1f} MW
            """
        )

st.divider()
st.markdown("*Built with ⚡ Streamlit — Electricity Demand Forecasting System*")
