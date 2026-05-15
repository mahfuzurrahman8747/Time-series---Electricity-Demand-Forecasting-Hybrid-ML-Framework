# 🌐 Streamlit Demo App — User Guide

Complete instructions for running the interactive electricity demand forecasting demo.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Ensure Models are Saved
Models should already be in `./models/` directory:
```
models/
├── gradient_boosting_model.pkl
├── xgboost_model.pkl
├── transformer_model.pkl
├── tcn_model.pkl
├── conformal_prediction_model.pkl
├── arima_model.pkl
├── sarima_model.pkl
├── standard_scaler.pkl
├── feature_names.json
└── model_metadata.json
```

If not present, run the `final_model.ipynb` notebook to generate them.

### Step 3: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 App Features

### 🎯 **Sidebar Controls**

#### Model Selection
- Choose from 7 pre-trained models
- Switch between them to compare predictions

#### Input Features
Configure real-time parameters:
- **Hour** (0-23) — Hour of the day
- **Day of Week** (0-6) — Monday to Sunday
- **Month** (1-12) — Month number
- **Temperature** (°C) — Current temperature
- **Load Shedding** (0/1) — Yes/No toggle
- **Wind Speed** (m/s) — Wind speed
- **Historical Lags** — Previous demand values

#### Advanced Settings
- **Lag Features**: Previous hour, day, and week demand
- **Rolling Statistics**: 24-hour mean and std dev
- **Day of Year**: Auto-calculated from month/day

---

## 🎮 How to Use

### Making a Prediction

1. **Adjust Sidebar Parameters**
   - Set time, weather, and historical values
   - Or use default values for a quick test

2. **Click "Generate Prediction" Button**
   - Model processes inputs
   - Returns predicted demand in MW

3. **View Results**
   - Prediction value displayed in green
   - Demand class (Low/Medium/High) shown
   - Model accuracy and MAE metrics displayed

### Comparing Models

1. **Change Model Selection**
   - Dropdown in sidebar
   - Automatically reloads selected model

2. **Generate New Prediction**
   - Same inputs with different model
   - Compare predictions

3. **View Model Performance**
   - Accuracy metric updates
   - MAE changes based on model
   - Performance comparison charts update

---

## 📈 Dashboard Sections

### 1. **Model Performance Metrics**
Top 4 cards showing:
- 📊 Current model accuracy
- 📌 MAE (Mean Absolute Error)
- 📐 R² Score
- 🎯 Model type (ML/Statistical/Deep Learning)

### 2. **Real-Time Prediction**
- Input summary display
- Prediction result (MW)
- Demand classification
- Model information

### 3. **Model Comparison**
Two side-by-side charts:
- Accuracy ranking across all models
- MAE comparison (lower is better)

### 4. **Feature Importance**
Bar chart showing which features impact predictions most

### 5. **24-Hour Forecast**
Line chart with confidence interval band (if available)

### 6. **Model Rankings Table**
Sortable table with:
- Model name
- Accuracy
- MAE
- R² score

---

## ⚙️ Configuration

### Modify Input Ranges
Edit sidebar number inputs in `streamlit_app.py`:

```python
hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
# Change value=12 to set different default
```

### Add Custom Models
1. Save new trained model to `./models/`
2. Update `load_all_models()` function
3. Add to sidebar model selection

### Change UI Theme
Modify CSS in app:
```python
st.markdown("""
<style>
    .stApp {
        background-color: #0F1923;  # Change background
    }
</style>
""", unsafe_allow_html=True)
```

---

## 🐛 Troubleshooting

### **Error: ModuleNotFoundError: No module named 'streamlit'**
```bash
pip install streamlit
```

### **Error: Failed to load models**
- Ensure `./models/` directory exists
- Check model files are present
- Run notebook to regenerate models:
```bash
jupyter notebook final_model.ipynb
# Run all cells
```

### **Error: No file found: 'gradient_boosting_model.pkl'**
Run this in terminal (from Research directory):
```bash
python -c "from pathlib import Path; Path('./models').mkdir(exist_ok=True)"
jupyter notebook final_model.ipynb
# Execute model saving cells
```

### **Port Already in Use (Address already in use)**
```bash
streamlit run streamlit_app.py --server.port 8502
```

### **App Loads Slowly**
- Models are large (~50-100 MB)
- First load caches models in memory
- Subsequent runs are faster
- Close other applications to free RAM

### **Prediction Returns NaN or Error**
- Check input ranges are valid
- Ensure scaler is compatible
- Try with default values first

---

## 📱 UI Tips & Tricks

### Keyboard Shortcuts
- **R** — Rerun script
- **C** — Clear cache (forces model reload)
- **K** — Toggle fullscreen
- **D** — Open/close documentation

### Dark Theme Benefits
- Easier on eyes during long sessions
- Better readability for presentations
- Professional appearance

### Interactive Charts
- **Hover** over bars to see values
- **Click legend** to toggle series
- **Drag** to zoom into charts
- **Double-click** to reset zoom

---

## 🔄 Update Models

To use newly trained models:

1. **Train new models** in `final_model.ipynb`
2. **Re-run model saving cells** (cells 2-3)
3. **Models auto-reload** in Streamlit (next run)

Or manually force reload:
```bash
# In terminal
streamlit run streamlit_app.py --logger.level=debug
# Press C to clear cache
```

---

## 📊 Expected Output Example

```
⚡ Electricity Demand Forecasting Dashboard

📈 Model Performance Summary
📊 Accuracy: 99.69%
📌 MAE (MW): 18.1
📐 R² Score: 0.9983
🎯 Model Type: ML Ensemble

🔮 Demand Prediction — Gradient Boosting
✅ Predicted Demand: 9,547.2 MW
Demand Class: 🟡 MEDIUM

Input Parameters:
• Time: 12:00 (Wednesday)
• Month: 5 | Temperature: 25.0°C
• Load Shedding: No ✅
• Wind Speed: 5.0 m/s

Model Used: gradient_boosting
Accuracy: 99.69%
```

---

## 🎨 Customization Ideas

### Add More Input Features
- Holidays flag
- Special events indicator
- Renewable energy generation

### Extend Forecasting
- 24-hour ahead forecast
- Weekly forecast
- Seasonal trends

### Add Analytics
- Feature sensitivity analysis
- Prediction confidence scores
- Demand pattern insights

### Improve UI
- Custom color scheme
- Map visualization
- Real-time data connections

---

## 📞 Support & Issues

**If you encounter problems:**

1. Check logs:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

2. Verify setup:
```bash
python -c "import streamlit, pandas, sklearn; print('✅ Setup OK')"
```

3. Recreate models:
```bash
jupyter nbconvert --to notebook --execute final_model.ipynb
```

---

## 🚀 Deployment

To deploy the app online:

### **Using Streamlit Cloud** (Free)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Deploy from your repo
4. Share public link

### **Using Docker** (Advanced)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "streamlit_app.py"]
```

### **Using Heroku** (Deprecated)
Alternative cloud platforms recommended.

---

## 💡 Best Practices

✅ **Do:**
- Use realistic input values (9000-12000 MW for demand)
- Check model accuracy before relying on predictions
- Compare multiple models for critical decisions
- Update models regularly with new data

❌ **Don't:**
- Use extreme input values outside training range
- Rely on single model for critical decisions
- Share raw model files without authentication
- Deploy without testing

---

**Happy Forecasting!** ⚡📊

Last Updated: May 2026

