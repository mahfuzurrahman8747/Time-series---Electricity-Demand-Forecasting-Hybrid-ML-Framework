# Research workflow

This guide follows the analysis implemented by the three core notebooks:
[`EDA.ipynb`](EDA.ipynb), [`Hyperparameter_tuning_of_models.ipynb`](Hyperparameter_tuning_of_models.ipynb),
and [`final_model.ipynb`](final_model.ipynb). It describes the current files
and does not replace the notebooks as the executable source of truth.

## 1. Prepare and explore the data

Run `EDA.ipynb` first.

1. Load the PGCB-derived demand and load-shedding CSV and the six regional
   hourly weather CSVs.
2. Select each region's timestamp, temperature, and wind speed; rename the
   weather columns by region; and outer-join the weather tables on timestamp.
3. Remove incomplete merged-weather timestamps, then calculate mean
   temperature and wind speed across the available regional columns.
4. Join the aggregated weather series to national demand and load shedding on
   timestamp and perform the notebook's cleaning and screening steps.
5. Export `merged_all_divisions_weather_data.csv`,
   `cleaned_combined_data.csv`, and `cleaned_min_hours_23.csv`, together with
   exploratory plots.

The checked-in prepared files contain 42,611 rows in
`cleaned_combined_data.csv` and 36,547 rows in `cleaned_min_hours_23.csv`.
The latter applies the notebook's minimum 23-observations-per-day filter.
These counts and date ranges should be rechecked after changing source data.

**Before running:** the notebook references raw CSVs using absolute local
paths. The raw inputs are outside `7_model/`; update the paths for your machine.
Run with `7_model/` as the working directory if generated exports should be
written here.

## 2. Engineer time-series features

Both modeling notebooks sort by timestamp and create:

- `hour_sin` and `hour_cos` from the hour-of-day cycle
- `lag1` and `lag23` from previous demand rows
- `temp_mean` and `wspd_mean` as exogenous inputs

The target is `demand_mw`. The resulting six-feature model matrix does not
include `load_shedding`; the final notebook uses that column for stress-period
analysis. Initial rows without lag history are dropped, leaving 36,524 samples
in the saved full-history training metadata.

The lag features are created with row shifts. Since the source can have
timestamp gaps, verify regular hourly spacing before interpreting `lag23` as
exactly 23 hours earlier.

## 3. Tune candidate configurations

Run `Hyperparameter_tuning_of_models.ipynb` after preparing the modeling data.

- **AutoARIMA:** compare three configurations at seasonal length 23. The
  recorded lowest RMSE is 234.456988 for the broadest tested configuration:
  `max_p=4`, `max_q=4`, `max_P=2`, `max_Q=2`, `max_order=6`, with
  approximation enabled.
- **XGBoost and Gradient Boosting:** use Optuna to minimize RMSE over 100
  trials. The validation set is the last 20% of the training segment, kept in
  chronological order.

The notebook currently loads its prepared CSV using an absolute path under
`5_model`. Point it to `7_model/cleaned_min_hours_23.csv` (or another intended
input) before execution. Tuning results and final-model settings are distinct:
the final notebook contains its own explicit model configurations.

## 4. Evaluate the forecasting models

Run `final_model.ipynb`. The notebook creates three cumulative datasets,
covering 2021-2023, 2021-2024, and 2021-2025. For each experiment, it reserves
the final six calendar months as a chronological test interval:

| Experiment | Test interval |
|---|---|
| 1 | 2023-07-01 through 2023-12-31 |
| 2 | 2024-07-01 through 2024-12-31 |
| 3 | 2025-07-01 through 2025-12-31 |

Gradient Boosting, XGBoost, and the residual ANN use the same six predictors
and a `StandardScaler` fitted on the corresponding training features. The
StatsForecast AutoARIMA model uses the time-series target and exogenous
regressors. The notebook computes MAE, RMSE, MAPE, normalized RMSE, R², and
bias, and includes analysis by load-shedding/stress conditions.

Review the saved results in `model_results_top.csv`. Results from distinct
test periods should remain identified by experiment; do not compare an
experiment's score as if all models were tested on the same dates.

## 5. Train and save reusable artifacts

The final notebook also trains full-history model instances and writes them
under `saved_models/`. Current outputs include:

- `M_GBoosting.pkl`, `M_XGBoost.pkl` / `M_XGBoost.json`, and `M_ANN.pth`
- `M_SF.pkl` for StatsForecast AutoARIMA
- `M_X_scaler.pkl` and `feature_name.json`
- `hourly_mean_demand.json` and `model_metadata.json`

The metadata records the full-history sample count, target, ordered features,
model configurations, and saved performance reference. Keep the feature
definition, scaler, and estimator in sync when updating the models.

## 6. Review figures and optional inference

Use `paper_figures/` to review exported EDA, model-comparison, forecast,
residual, and feature-impact figures. `Publication-quality.ipynb` prepares
publication-oriented figure outputs.

The optional `streamlit_app.py` uses saved artifacts for inference; it does
not train models. Follow `STREAMLIT_GUIDE.md` for its inputs and limitations.
For long-range recursive forecasts, future lag values are predictions rather
than observations. Use weather and operational inputs that would actually be
available at the forecast issue time.

## Reproducibility checklist

- Confirm the selected source and prepared CSV paths before running a notebook.
- Record the dataset date range, row count, and timestamp gaps after cleaning.
- Keep train/test boundaries chronological and do not fit scalers on test rows.
- Report experiment number and exact evaluation interval with every metric.
- Record Python and package versions when generating new tuning results or
  artifacts.
