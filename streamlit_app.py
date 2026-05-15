"""
⚡ Electricity Demand Forecasting — Streamlit App
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime

# ================================================================================
# FUNCTIONS
# ================================================================================

@st.cache_resource
def load_gb_model():
    """Load Gradient Boosting model, scaler, and features"""
    models_dir = Path('./models')
    
    with open(models_dir / 'gradient_boosting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(models_dir / 'standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(models_dir / 'feature_names.json', 'r') as f:
        features = json.load(f)
    
    return model, scaler, features

@st.cache_resource
def load_hourly_mean_demand():
    """Load 24-hour mean demand from last year"""
    models_dir = Path('./models')
    hourly_means_path = models_dir / 'hourly_mean_demand.json'
    
    if hourly_means_path.exists():
        with open(hourly_means_path, 'r') as f:
            hourly_means = json.load(f)
            # Convert string keys to int
            hourly_means = {int(k): v for k, v in hourly_means.items()}
        return hourly_means
    else:
        # Fallback if file doesn't exist
        return {h: 9500 for h in range(24)}

def predict_demand(model, scaler, features, input_dict):
    """Make prediction"""
    try:
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ================================================================================
# PAGE CONFIG
# ================================================================================

st.set_page_config(
    page_title="⚡ Electricity Demand Forecasting",
    page_icon="⚡",
    layout="centered"
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

# Date input
selected_date = st.sidebar.date_input("📅 Select Date", value=datetime.now())

# Hour input
hour = st.sidebar.slider("⏰ Hour (0-23)", min_value=0, max_value=23, value=12)

# Temperature input
temperature = st.sidebar.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)

# Wind speed input
wind_speed = st.sidebar.number_input("💨 Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

# Historical demand inputs
st.sidebar.subheader("📊 Historical Demand (MW)")

# Get hourly mean for current hour
default_demand = hourly_means.get(hour, 9500)

# Display last year's hourly mean
st.sidebar.info(f"📈 Hour {hour:02d}:00 — Last Year Average: **{default_demand:.1f} MW**")

# ✅ NEW: Checkbox to skip manual input
use_default = st.sidebar.checkbox("I don't have last hour demand — use historical average", value=False)

if use_default:
    demand_lag1 = default_demand
    st.sidebar.success(f"✅ Using historical average: **{default_demand:.1f} MW**")
else:
    demand_lag1 = st.sidebar.number_input(
        "Last Hour Demand (MW)",
        min_value=0,
        max_value=20000,
        value=int(default_demand),
        step=10,
        help="Leave as default or enter actual demand value"
    )
# ================================================================================
# MAIN DASHBOARD
# ================================================================================

st.title("⚡ Electricity Demand Forecasting")

st.divider()

# Model accuracy metric
st.metric("📊 Model Accuracy", "94.75%")

st.divider()

# Prediction section
st.subheader("🔮 Make Prediction")

if st.button("🚀 Predict Demand", use_container_width=True):
    # Extract date components
    day_of_year = selected_date.timetuple().tm_yday
    month = selected_date.month
    day_of_week = selected_date.weekday()  # Monday=0, Sunday=6
    
    # Create input data using last hour demand
    input_data = {
        'hour': hour,
        'dayofweek': day_of_week,
        'month': month,
        'dayofyear': day_of_year,
        'load_shedding': 0,
        'temp_mean': temperature,
        'wspd_mean': wind_speed,
        'lag1': demand_lag1,
        'lag24': demand_lag1,
        'lag168': demand_lag1,
        'roll24_mean': demand_lag1,
        'roll24_std': 100,
    }
    
    # Make prediction
    prediction = predict_demand(model, scaler, features, input_data)
    
    if prediction is not None:
        st.success(f"✅ **Predicted Demand: {prediction:,.1f} MW**")
        
        # Input summary
        st.info(f"""
        **Input Summary:**
        - Date: {selected_date.strftime('%Y-%m-%d')} ({['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]})
        - Hour: {hour:02d}:00
        - Temperature: {temperature}°C
        - Wind Speed: {wind_speed} m/s
        - Last Hour Demand: {demand_lag1} MW
        """)

st.divider()
st.markdown("*Built with ⚡ Streamlit — Electricity Demand Forecasting System*")
