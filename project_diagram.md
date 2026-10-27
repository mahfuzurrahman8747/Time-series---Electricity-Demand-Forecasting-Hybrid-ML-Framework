# Project workflow diagram

This diagram summarizes the implemented data preparation, tuning, evaluation,
artifact, and optional inference paths for the `7_model` project.

```mermaid
flowchart TD
    subgraph SOURCES["Source data"]
        DEMAND["PGCB-derived demand and load-shedding CSV"]
        WEATHER["Six regional hourly weather CSVs"]
    end

    subgraph EDA["EDA.ipynb"]
        READ["Load, inspect, and clean sources"]
        JOIN["Outer-join regional weather by timestamp"]
        FILTER["Remove incomplete merged-weather timestamps"]
        AGG["Calculate mean temperature and wind speed"]
        MERGE["Join weather means to demand and load shedding"]
        PREP["Clean, screen, and apply daily completeness filter"]
        OUTPUTS["Export merged_all_divisions_weather_data.csv<br/>cleaned_combined_data.csv<br/>cleaned_min_hours_23.csv"]
        FIGS["Export exploratory figures"]
        READ --> JOIN --> FILTER --> AGG --> MERGE --> PREP --> OUTPUTS
        PREP --> FIGS
    end

    DEMAND --> READ
    WEATHER --> READ

    subgraph TUNING["Hyperparameter_tuning_of_models.ipynb"]
        GRID["Compare three AutoARIMA configurations"]
        OPTUNA["Optuna search: Gradient Boosting and XGBoost<br/>time-ordered validation tail"]
        TUNING_OUT["Tuned candidate configurations"]
        GRID --> TUNING_OUT
        OPTUNA --> TUNING_OUT
    end

    OUTPUTS --> FEATURES["final_model.ipynb<br/>Cyclic hour features + temp/wind means<br/>lag1 + lag23; target = demand_mw"]
    FEATURES --> SPLITS["Three cumulative time-based experiments<br/>final six months held out per experiment"]
    TUNING_OUT -. "configuration reference" .-> SPLITS

    subgraph MODELS["Model fitting and evaluation"]
        SPLITS --> FIT["Gradient Boosting<br/>XGBoost<br/>Residual ANN<br/>AutoARIMA with exogenous inputs"]
        FIT --> METRICS["MAE, RMSE, MAPE, normalized RMSE,<br/>R², bias, and stress analysis"]
        METRICS --> RESULTS["model_results_top.csv<br/>comparison figures"]
        FIT --> FULL["Train full-history model artifacts"]
    end

    subgraph DELIVERABLES["Outputs"]
        FULL --> SAVED["saved_models/<br/>estimators, ANN weights, scaler,<br/>feature names, metadata"]
        METRICS --> DIAGNOSTICS["paper_figures/<br/>forecast, residual, comparison,<br/>EDA, and feature-impact plots"]
        SAVED --> APP["Optional streamlit_app.py<br/>loads artifacts for inference"]
    end

    classDef source fill:#e8f1ff,stroke:#4472c4,color:#172b4d
    classDef process fill:#e9f7ef,stroke:#3a8f5b,color:#173b25
    classDef model fill:#fff2cc,stroke:#c99a00,color:#443400
    classDef output fill:#f1e8ff,stroke:#8054a6,color:#30203c
    class DEMAND,WEATHER source
    class READ,JOIN,FILTER,AGG,MERGE,PREP,FEATURES,SPLITS,GRID,OPTUNA process
    class FIT,FULL,APP model
    class OUTPUTS,FIGS,TUNING_OUT,METRICS,RESULTS,SAVED,DIAGNOSTICS output
```

## Interpretation notes

- The final model uses six predictors: `hour_sin`, `hour_cos`, `temp_mean`,
  `wspd_mean`, `lag1`, and `lag23`. `load_shedding` is used for stress analysis,
  not as a model predictor.
- Lags are shifted by row, and days with 23 observations may still contain
  missing hourly timestamps. Confirm cadence before interpreting a row lag as
  a fixed elapsed-time lag.
- The three reported test periods are July-December 2023, 2024, and 2025.
- Tuning results inform candidate selection; they are not automatically the
  same as the fixed configurations used by the final notebook.
- The raw source CSVs are external to this folder, and notebook cells include
  machine-specific paths that may need updating.
