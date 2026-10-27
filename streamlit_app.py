"""Streamlit interface for the electricity demand forecasting models."""

from datetime import datetime, time

import numpy as np
import pandas as pd
import streamlit as st

from forecast_engine import (
    MAX_FORECAST_HOURS,
    MODEL_PATHS,
    load_forecast_model,
    load_hourly_mean_demand,
    load_sarima_model,
    load_validation_r2,
    next_forecast_time,
    predict_demand_at,
    predict_ml_demand_at,
)


st.set_page_config(
    page_title="Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Electricity Demand Forecasting")
st.caption("Enter the forecast time and recent conditions to estimate electricity demand.")

st.sidebar.header("⚙️ Input parameters")
selected_model_name = st.sidebar.selectbox(
    "Choose model",
    options=list(MODEL_PATHS),
    index=0,
)

try:
    history_model = load_sarima_model()
    selected_model, feature_scaler, feature_names = load_forecast_model(
        selected_model_name
    )
    hourly_means = load_hourly_mean_demand()
    first_forecast_time = next_forecast_time(history_model)
except Exception as error:
    st.error(f"Could not load forecast inputs or the selected model: {error}")
    st.stop()

last_training_time = pd.Timestamp(history_model.last_dates[0])
st.metric(
    f"{selected_model_name} latest test R²",
    f"{load_validation_r2(selected_model_name):.2%}",
)
st.caption(f"Forecast history through {last_training_time:%Y-%m-%d %H:%M}")
st.caption(
    "The saved non-AutoARIMA model artifacts do not record their own training "
    "cutoff; their recursive demand lags are seeded from the shared saved history."
)

default_forecast_time = max(
    pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1),
    first_forecast_time,
)
forecast_date = st.sidebar.date_input(
    "Select date",
    value=default_forecast_time.date(),
    min_value=first_forecast_time.date(),
    max_value=(last_training_time + pd.Timedelta(hours=MAX_FORECAST_HOURS)).date(),
)
forecast_hour = st.sidebar.select_slider(
    "Hour (0-23)",
    options=list(range(24)),
    value=default_forecast_time.hour,
    format_func=lambda hour: f"{hour:02d}:00",
)

temperature = st.sidebar.number_input(
    "🌡️ Temperature (°C)", value=25.0, step=0.1
)
wind_speed = st.sidebar.number_input(
    "💨 Wind speed (m/s)", min_value=0.0, value=5.0, step=0.1
)
previous_hour = (forecast_hour - 1) % 24
historical_average = hourly_means[previous_hour]
st.sidebar.subheader("📊 Historical demand (MW)")
st.sidebar.info(
    f"Hourly mean for {previous_hour:02d}:00: {historical_average:,.2f} MW"
)
use_historical_average = st.sidebar.checkbox(
    "I don't have last hour demand — use historical average",
    value=True,
)
if use_historical_average:
    last_hour_demand = historical_average
    st.sidebar.success(f"Using historical average: {last_hour_demand:,.1f} MW")
else:
    last_hour_demand = st.sidebar.number_input(
        "Last hour demand (MW)",
        min_value=0.0,
        value=historical_average,
        step=100.0,
        format="%.1f",
    )

st.divider()
st.header("🔮 Make prediction")
st.caption(
    "The forecast horizon counts hourly steps from the available model history "
    "through the selected time. Intermediate hours are forecast recursively "
    "because demand-lag features use earlier actual or forecast demand."
)

if st.button("🚀 Predict demand", type="primary", width="stretch"):
    selected_forecast_time = datetime.combine(
        forecast_date,
        time(hour=forecast_hour),
    )
    progress = None
    try:
        progress = st.progress(
            0,
            text="Starting recursive forecast; longer horizons need more iterations.",
        )
        with st.spinner("Calculating the forecast up to the selected hour..."):
            if selected_model_name == "AutoARIMA (StatsForecast)":
                target_forecast, horizon, target_inputs = predict_demand_at(
                    selected_model,
                    selected_forecast_time,
                    temperature,
                    wind_speed,
                    last_hour_demand,
                    progress,
                )
                forecast_name = selected_model.models[0].alias
                prediction = float(target_forecast[forecast_name])
                prediction_intervals = {
                    level: (
                        float(target_forecast[f"{forecast_name}-lo-{level}"]),
                        float(target_forecast[f"{forecast_name}-hi-{level}"]),
                    )
                    for level in (80, 90)
                }
                target_lag1 = float(target_inputs["lag1"])
            else:
                if selected_forecast_time < first_forecast_time:
                    raise ValueError(
                        "Choose a forecast time after the available model history "
                        f"({last_training_time:%Y-%m-%d %H:%M})."
                    )
                horizon = (
                    int(
                        (selected_forecast_time - first_forecast_time)
                        / pd.Timedelta(hours=1)
                    )
                    + 1
                )
                if horizon > MAX_FORECAST_HOURS:
                    raise ValueError(
                        "The selected time is more than one year beyond the "
                        "available model history. Choose a closer date."
                    )
                forecast_times = pd.date_range(
                    first_forecast_time,
                    periods=horizon,
                    freq="h",
                )
                prediction, horizon = predict_ml_demand_at(
                    selected_model_name,
                    selected_model,
                    feature_scaler,
                    feature_names,
                    history_model.ga.data,
                    forecast_times,
                    temperature,
                    wind_speed,
                    last_hour_demand,
                    progress,
                )
                prediction_intervals = None
                target_lag1 = float(last_hour_demand)
    except Exception as error:
        if progress is not None:
            progress.empty()
        st.error(f"{selected_model_name} prediction failed: {error}")
    else:
        progress.empty()
        if prediction_intervals is not None and not np.isfinite(
            [bound for interval in prediction_intervals.values() for bound in interval]
        ).all():
            st.error("The model returned a non-finite prediction interval.")
            st.stop()

        st.markdown(
            """
<style>
.st-key-forecast-results [data-testid="stMetricValue"] {
    font-size: clamp(2.4rem, 4vw, 3.6rem);
    font-weight: 700;
    line-height: 1.15;
}
</style>
""",
            unsafe_allow_html=True,
        )
        with st.container(key="forecast-results"):
            st.metric(
                f"{selected_model_name} predicted demand",
                f"{prediction:,.1f} MW",
                border=True,
            )
            if prediction_intervals is not None:
                with st.container(horizontal=True):
                    for level, (lower, upper) in prediction_intervals.items():
                        st.metric(
                            f"{level}% prediction interval",
                            f"{lower:,.1f}–{upper:,.1f} MW",
                            border=True,
                        )
                st.caption(
                    "AutoARIMA model-based intervals at the selected hour. They "
                    "are conditional on the entered weather and recursive demand "
                    "inputs."
                )
            else:
                st.info(
                    "The selected model artifact does not provide calibrated "
                    "prediction intervals. Choose AutoARIMA to see the 80% and "
                    "90% intervals."
                )
        with st.container(border=True):
            st.subheader("Input summary")
            st.markdown(
                f"""
- **Date:** {selected_forecast_time:%Y-%m-%d (%A)}
- **Hour:** {selected_forecast_time:%H:%M}
- **Temperature:** {temperature:.1f} °C
- **Wind speed:** {wind_speed:.1f} m/s
- **Last hour demand:** {target_lag1:,.1f} MW
- **Forecast horizon:** {horizon:,} hourly steps (~{horizon / 24:.1f} days) from the training cutoff
"""
            )

st.divider()
st.caption(f"Built with Streamlit · Selected model: {selected_model_name}")
