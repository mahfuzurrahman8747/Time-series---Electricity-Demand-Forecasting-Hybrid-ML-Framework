# 🌐 Streamlit Demo – Interactive Electricity Demand Forecast

This guide covers the **Streamlit** app currently implemented in `streamlit_app.py`.
It loads the saved Gradient Boosting model and lets you enter temporal, weather,
and demand inputs to generate a single electricity-demand prediction in MW.

---

## 📦 Prerequisites
| Item | Why? |
|------|------|
| Python 3.8+ | Required by the notebook and Streamlit runtime |
| `requirements.txt` | Pinpointed library versions for reproducibility |
| Trained model artefacts (`saved_models/`) | The app loads these at start‑up |

If you haven't installed the dependencies yet, run:
```bash
pip install -r requirements.txt
```

## 📂 Directory Layout (relevant parts)
```
electricity-demand-forecasting/
├─ streamlit_app.py            # Main Streamlit entry point
├─ saved_models/               # Serialized models & scalers
│   ├─ M_ANN.pth
│   ├─ M_XGBoost.pkl
│   └─ ...
├─ paper_figures/              # High‑resolution plots used in docs
└─ ...
```

## 🚀 Launch the Dashboard
```bash
streamlit run streamlit_app.py
```
The app will open at **http://localhost:8501** (or another free port if 8501 is taken).  If you need a specific port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 🖥️ UI Overview
### Sidebar Controls
- **Temporal input** – Select a date and hour.
- **Weather inputs** – Enter temperature (°C) and wind speed (m/s).
- **Demand input** – Enter last-hour demand (MW), or check the box to use the
  historical-average fallback. If `saved_models/hourly_mean_demand.json` is
  absent, the app uses 9,500 MW for every hour.

### Main Panel
- The app displays a fixed model-accuracy label and, after prediction, the
  predicted demand (MW) and a summary of the selected inputs.
- The current app does not provide a model-selection dropdown, demand class,
  feature-importance chart, or 24-hour forecast chart.
- The inference feature row uses the supplied last-hour demand value for all
  demand lags and rolling means, sets `load_shedding` to zero, and estimates
  rolling standard deviations from that demand value. Use this as a demo
  workflow, not as a substitute for supplying the trained model's complete,
  correctly aligned historical feature window.

The model-comparison images in `paper_figures/` are offline evaluation outputs,
not panels rendered by the current app.

---

## ⚙️ Customising the App
### Adding a New Model
The app currently loads only `M_GBoosting.pkl` through `load_gb_model()`; there
is no model-selection dropdown or `load_all_models()` function. Supporting a
different model requires updating the loader, compatible scaler and ordered
feature list, feature construction, and prediction UI together.

### Tweaking the UI Theme
The current app does not define custom CSS. Configure the Streamlit theme in
`.streamlit/config.toml` if a different appearance is needed.

### Changing Input Ranges
Modify the corresponding `st.sidebar` input in `streamlit_app.py`. For example,
to widen the temperature range:
```python
temperature = st.sidebar.number_input(
    "🌡️ Temperature (°C)",
    min_value=-30.0,
    max_value=50.0,
    value=25.0,
    step=0.1,
)
```

---

## 🐛 Troubleshooting
| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: streamlit` | `pip install streamlit` |
| Model files missing | Ensure `saved_models/` contains the .pkl/.pth files; re‑run `final_model.ipynb` to generate them. |
| Port already in use | Start with a different port: `streamlit run streamlit_app.py --server.port 8502` |
| Predictions return `NaN` | Check that all inputs are within the training range (see the *Input Ranges* section above). |

---

## 📦 Deployment Options
### Streamlit Cloud (quick, free)
1. Push the repository to GitHub.
2. Sign‑in at https://share.streamlit.io and select the repo.
3. Set the entry‑point to `streamlit_app.py`.
4. Deploy – a public URL will be generated.

### Docker (self‑hosted)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
```
Build & run:
```bash
docker build -t demand‑forecast .
docker run -p 8501:8501 demand‑forecast
```

### Heroku (deprecated) – see Streamlit Cloud instead.

---

## 📚 Further Reading & References
- **Conformal Prediction** – https://arxiv.org/abs/1904.06857
- **Time‑Series Forecasting** – Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*
- **XGBoost Documentation** – https://xgboost.readthedocs.io/

---

## 🤝 Contributing
Feel free to open issues or pull requests.  Typical contributions include:
- Adding new weather or calendar features
- Implementing a true deep‑learning model (LSTM, Prophet, etc.)
- Improving visualisations (interactive Plotly charts, map overlays)

---

## 📧 Support
If you run into problems, try the troubleshooting table above.  For anything else, reach out via:
- Email: mahfuzurrahman8747@gmail.com
- GitHub Issues: https://github.com/yourusername/electricity-demand-forecasting/issues

---

**Happy forecasting!** ⚡📊

*Last updated: September 2026*