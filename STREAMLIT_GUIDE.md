# Streamlit forecasting guide

This guide describes the interactive demand-forecasting interface in
[`streamlit_app.py`](streamlit_app.py). It is an inference application: it loads
already fitted models and preprocessing artifacts from `saved_models/` and
does not train or update models when a user requests a prediction.

## Run the app

From the `6_model` directory, activate the Python environment containing the
project dependencies, then run:

```bash
python -m streamlit run streamlit_app.py
```

If the project's virtual environment is `Research/venv`, use this PowerShell
command from `6_model`:

```powershell
..\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The app is normally available at `http://localhost:8501`. To use another port:

```bash
python -m streamlit run streamlit_app.py --server.port 8502
```

Install the dashboard dependencies if needed:

```bash
python -m pip install numpy pandas scikit-learn joblib statsforecast xgboost torch streamlit
```

## Choose a model and enter inputs

The sidebar offers four saved models:

| Dashboard option | Artifact | Inference behavior |
|---|---|---|
| AutoARIMA (StatsForecast) | `saved_models/M_SF.pkl` | Uses the saved StatsForecast model and native 80% and 90% intervals |
| Gradient Boosting | `saved_models/M_GBoosting.pkl` | Loads a fitted estimator and applies the saved feature scaler |
| XGBoost | `saved_models/M_XGBoost.pkl` | Loads a fitted estimator and applies the saved feature scaler |
| Residual ANN | `saved_models/M_ANN.pth` | Reconstructs the matching network, loads its saved weights, and performs inference |

The user selects a date and hour and enters temperature, wind speed, and the
load-shedding feature. For previous-hour demand, the app can use the matching
hour-of-day value from `saved_models/hourly_mean_demand.json`, or the user can
enter a demand value in MW.

For example, a forecast at 00:00 uses the JSON mean for hour 23 as the default
previous-hour demand. This value is used as the final forecast hour's `lag1`;
future intermediate lag values are generated recursively.

## How inference works

The app loads the ordered seven-feature list from
`saved_models/feature_name.json`:

```text
hour_sin, hour_cos, load_shedding, temp_mean, wspd_mean, lag1, lag23
```

The same fitted scaler in `saved_models/M_X_scaler.pkl` is used for the
machine-learning models. The scaler and feature order must match those used
when the estimators and ANN weights were trained.

The app forecasts hourly from the available model history to the selected
timestamp. It builds calendar features for each hour and feeds forecast demand
back into future `lag1` and `lag23` features. Temperature, wind speed, and
load-shedding inputs are held constant across these intermediate hours.

### ANN uses saved weights only

The Residual ANN option loads `saved_models/M_ANN.pth` as a weights-only
state dictionary, maps it to CPU, restores it into the matching network
architecture, and sets the model to evaluation mode. The forecast runs under
PyTorch inference mode. No training, optimizer step, or weight update occurs
in Streamlit. To create new ANN weights, run the training workflow in
`final_model.ipynb` separately and save the resulting state dictionary.

## Prediction intervals and scores

- AutoARIMA displays model-based 80% and 90% prediction intervals for the
  selected hour, conditional on the entered weather and recursive demand
  inputs.
- The Gradient Boosting, XGBoost, and ANN artifacts do not contain calibrated
  prediction intervals. The app intentionally does not invent interval ranges
  for those models.
- The displayed validation R² is read from `model_results_top.csv`. It is a
  saved evaluation statistic, not a score calculated from the user's live
  forecast and not classification accuracy.

## Required saved files

The dashboard expects these files to exist:

| File | Used for |
|---|---|
| `saved_models/M_SF.pkl` | AutoARIMA model and the common saved demand history |
| `saved_models/M_GBoosting.pkl` | Gradient Boosting estimator |
| `saved_models/M_XGBoost.pkl` | XGBoost estimator |
| `saved_models/M_ANN.pth` | Residual ANN weights |
| `saved_models/M_X_scaler.pkl` | Scaler for the machine-learning model inputs |
| `saved_models/feature_name.json` | Model input feature names and order |
| `saved_models/hourly_mean_demand.json` | Hourly fallback demand values |
| `model_results_top.csv` | Validation R² values shown in the interface |

If a required artifact is missing or cannot be loaded, the app reports the
loading error instead of silently substituting another model.

## Forecast horizon and performance

The horizon is the number of hourly steps between the saved model history and
the selected timestamp, including every intermediate hour. It is not just the
number of hours beyond the current time.

All model options recursively forecast intermediate hours because demand lags
are inputs. This means a date far beyond the saved training history requires
many predictions. AutoARIMA also repeats the full-horizon forecast while
updating lag estimates until they converge (up to the configured iteration
limit). The progress indicator shows the work being performed. The app limits
the forecast to one year beyond the saved history.

## Known limitations

- The saved non-AutoARIMA model artifacts do not record their individual
  training cutoff. Their recursive demand history is initialized from the
  available saved AutoARIMA history, and the app displays this caveat.
- Forecast weather and load-shedding values are assumed constant for all
  intermediate hours.
- Forecast demand lags after the saved history are predictions, not observed
  demand. Long-range results therefore accumulate recursive uncertainty.
- The supplied hourly mean is a fallback for the final forecast hour's
  previous-hour demand; it is not a replacement for the complete historical
  lag sequence.
- The saved machine-learning models and scaler must be regenerated together if
  the feature definitions or order change.

For the overall project, evaluation table, notebook workflow, and figure
gallery, see [`README.md`](README.md).
