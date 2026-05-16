# 🚀 Getting Started Guide

Quick setup instructions to get the electricity demand forecasting project running.

---

## ⚡ TL;DR (5 Minutes)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/electricity-demand-forecasting.git
cd electricity-demand-forecasting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook final_model.ipynb

# 5. (Later) Run Streamlit app
streamlit run streamlit_app.py
```

---

## 📦 Full Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/electricity-demand-forecasting.git
cd electricity-demand-forecasting
```

### Step 2: Set Up Virtual Environment

#### Option A: Using `venv` (Built-in)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Verify activation (should show (venv) prefix)
python --version
```

#### Option B: Using `conda`
```bash
# Create environment
conda create -n demand-forecast python=3.9

# Activate
conda activate demand-forecast
```

### Step 3: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn xgboost matplotlib jupyter notebook streamlit
```

### Step 4: Verify Installation

```bash
python -c "import pandas, sklearn, xgboost; print('✅ All libraries loaded successfully!')"
```

---

## 📓 Running the Jupyter Notebook

### Start Jupyter

```bash
jupyter notebook
```

Your browser will open at `http://localhost:8888`

### Open the Notebook

1. Click on `final_model.ipynb`
2. Run cells sequentially (or use `Ctrl+A` → `Shift+Enter` to run all)
3. Wait for completion (approximately 5-10 minutes depending on your hardware)

### Expected Outputs

After running, you'll get:

```
✅ model_results_top_8.csv
✅ model_results_top_8.png
✅ classification_metrics_top_8.csv
✅ classification_metrics_dashboard_top_8.png
✅ feature_importance_ranking_top_8.csv
✅ feature_importance_by_model.png
✅ feature_importance_aggregated_top_8.png
```

---

## 🌐 Running the Streamlit App (Coming Soon)

Once you have the app ready:

```bash
# Make sure you're in the project directory with streamlit_app.py
streamlit run streamlit_app.py
```

**What you'll see:**
- 🎯 Real-time demand predictions
- 📊 Model performance comparison
- 📈 24-hour forecast charts
- 🎚️ Interactive feature sliders
- 📋 Model rankings and metrics

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
# Make sure virtual environment is activated, then:
pip install -r requirements.txt
```

### Issue: Jupyter not found

**Solution:**
```bash
pip install jupyter notebook
jupyter notebook
```

### Issue: XGBoost import error

**Solution:**
```bash
pip install --upgrade xgboost
```

### Issue: Data file not found

**Solution:**
Make sure `cleaned_combined_data.csv` is in the same directory as `final_model.ipynb`, or update the path in the notebook:

```python
# In cell 3, change:
df = pd.read_csv('path/to/cleaned_combined_data.csv')
```

### Issue: Out of Memory error

**Solution:**
- Close other applications
- Consider using a smaller sample of data for testing
- Increase swap space if on Linux/Mac

---

## 📊 Data Requirements

The project expects these files:

```
cleaned_combined_data.csv         # Main dataset
```

**CSV Format:**
```
datetime,demand_mw,load_shedding,temp_mean,wspd_mean
2020-01-01 00:00:00,8500,0,22.5,3.2
2020-01-01 01:00:00,8200,0,21.8,3.1
...
```

---

## 🎯 Project Workflow

```
1. Load data
   ↓
2. Feature engineering (lags, rolling stats, temporal)
   ↓
3. Train/test split (80/20)
   ↓
4. Train 7 models in parallel
   ↓
5. Evaluate metrics (MAE, RMSE, R², Accuracy)
   ↓
6. Extract feature importance
   ↓
7. Generate visualizations
   ↓
8. Save results to CSV/PNG
   ↓
9. Display top 5 models summary
```

---

## 📚 Key Files Explained

| File | Purpose |
|------|---------|
| `final_model.ipynb` | Main training notebook |
| `streamlit_app.py` | Interactive demo app (template) |
| `requirements.txt` | Python dependencies |
| `cleaned_combined_data.csv` | Input dataset |
| `model_results_top_8.csv` | Model performance metrics |
| `model_results_top_8.png` | Performance visualization |

---

## 🚀 Next Steps

1. ✅ **Complete Installation** — Follow steps above
2. 📓 **Run Notebook** — Execute `final_model.ipynb`
3. 📊 **Review Results** — Check generated CSV/PNG files
4. 🌐 **Explore Visualizations** — View charts and metrics
5. 🔮 **Customize Models** — Modify hyperparameters
6. 🎨 **Build Streamlit App** — Use template to create demo

---

## 💡 Tips & Best Practices

### For Better Performance

```python
# In the notebook, you can modify:

# Increase XGBoost complexity for better accuracy
m_xgb = xgboost.XGBRegressor(
    n_estimators=500,      # More trees
    learning_rate=0.01,    # Slower learning
    max_depth=7,           # Deeper trees
    random_state=42
)

# Adjust feature lags for different patterns
ATTN_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]  # Add weekly lag
```

### For Faster Testing

```python
# Use a sample of data for testing:
df_sample = df.sample(frac=0.1, random_state=42)
# Continue with df_sample instead of df
```

### For Custom Visualizations

```python
# Modify chart colors
CAT_COLOR = {
    'XGBoost': '#FF6B6B',      # Red
    'Transformer': '#4ECDC4',   # Teal
    # ... add more
}
```

---

## 📞 Getting Help

**If you encounter issues:**

1. Check [Troubleshooting](#troubleshooting) section above
2. Review [README.md](README.md) for detailed documentation
3. Open an issue on GitHub with:
   - Error message (full traceback)
   - Your Python version
   - Your OS (Windows/Mac/Linux)
   - Steps to reproduce

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip list | grep pandas` should work)
- [ ] Jupyter runs without errors
- [ ] Notebook loads data successfully
- [ ] First model trains and completes
- [ ] CSV results file created
- [ ] PNG visualization generated

---

## 🎉 You're Ready!

Once verification complete, you're set to explore the forecasting models and generate predictions!

**Happy Forecasting!** ⚡📊

