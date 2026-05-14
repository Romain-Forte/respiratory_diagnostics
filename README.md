# Respiratory Diagnostics

Toolkit for preparing EFRAIM/BAZEA clinical data, engineering reproducible respiratory-diagnostic features, training evaluation pipelines, and serving saved `joblib` models for single-patient inference.

## Architecture

```text
respiratory_diagnostics/
  analyse_data/                  Intermediate tables and exploratory notebooks
  api/                           FastAPI wrapper for saved-model inference
  configs/                       Per-diagnostic config files for model selection
  models/                        Saved `*.joblib` inference artifacts
  tests/                         Regression tests for saved-model inference
  utils/                         Reusable data, modeling, validation, and inference modules
  bazea_dataset.ipynb            Main notebook for end-to-end experiments
  dev_algo.ipynb                 Model development and comparison notebook
  README.md
```

The repository has two main runtime paths:

1. Feature engineering and model development from notebooks/scripts.
2. Saved-model inference through `utils.saved_model_inference` or the HTTP API in `api/`.

## Core Modules

- `utils/feature_loader.py` and `utils/feature_columns.json`
  Load curated feature subsets used by notebooks and pipelines.
- `utils/feature_transformer.py`
  Applies domain transformations, regroupings, scaling helpers, and feature cleaning.
- `utils/algo_prediction.py`
  Contains training-oriented helpers such as train/test preparation and `AutoStandardScaler`.
- `utils/run_model.py`
  Orchestrates training, augmentation, thresholding, sensitivity analysis, and config-driven runs.
- `utils/model_saving.py`
  Persists trained pipelines into the `models/` directory as `joblib` payloads.
- `utils/saved_model_inference.py`
  Loads saved pipelines and runs one-row inference across all saved diagnostics.
- `api/saved_model_inference_app.py`
  Exposes the saved-model inference flow over FastAPI.

## Saved-Model Inference Flow

`utils/saved_model_inference.py` is structured as a small inference pipeline:

1. Discover all `*.joblib` files from `models/`.
2. Load each model once and extract:
   diagnostic name, predictor pipeline, expected feature columns, and threshold.
3. Build a one-row `DataFrame` from incoming `feature_values`.
4. Align that row to each model's expected columns.
5. Use `predict_proba` when available, otherwise fallback to `predict`.
6. Return a dictionary keyed by diagnostic name.

Each result entry contains:

- `diagnostic`
- `model_path`
- `probability`
- `threshold`
- `prediction`

Important implementation note:

- saved pipelines are not standalone binary blobs; they depend on Python classes available at load time, notably `AutoStandardScaler` from `utils.algo_prediction`.

## Quick Start

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For the API:

```powershell
pip install -r requirements-api.txt
```

Some research notebooks also expect optional packages such as `shap`, `xgboost`, `lightgbm`, `catboost`, `scikit-multilearn`, or `tabpfn`.

### 2. Prepare data

Place cleaned EFRAIM/BAZEA exports in `analyse_data/` and load them from a notebook or script.

### 3. Engineer features

```python
import pandas as pd
from utils.feature_loader import load_columns
from utils.feature_transformer import transform_features

raw = pd.read_excel("analyse_data/efraim_export.xlsx")
selected = load_columns("utils/feature_columns.json", sections=["clinical_core"])
features = transform_features(raw[selected])
```

### 4. Train or evaluate models

```python
from utils.algo_prediction import preparer_jeu_xy
from utils.models_and_metrics import get_models

y = raw[[c for c in raw.columns if c.startswith("Etiology_")]]
X_train, X_test, y_train, y_test, labels = preparer_jeu_xy(features, y)
model = get_models(y_train)["Random Forest"]
```

For more complete experiment flows, use `utils/run_model.py` or the notebooks.

## Programmatic Inference

```python
from utils.saved_model_inference import predict_all_saved_models

results = predict_all_saved_models(
    feature_values=[...],
    feature_names=[...],  # optional if the models can infer the expected columns
)
```

Validation rules enforced by the inference layer:

- `feature_names` must be non-empty strings and unique.
- `feature_values` and `feature_names` must have the same length.
- all columns required by each model must be present in the input row.
- each saved model must expose `pipe_inference` or `pipe_train`.

## HTTP API

Run the FastAPI wrapper locally:

```powershell
uvicorn api.saved_model_inference_app:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`

Example payload:

```json
{
  "feature_values": [0.98, 0, 0.98],
  "feature_names": ["Time H-ICU", "TIME SYMPTOMES-ICU", "Time  DG-ICU"]
}
```

The repository also includes a Docker entrypoint targeting the same API.

## Testing

Targeted regression test for the saved-model inference path:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_saved_model_inference -v
```

This test suite covers:

- inference across multiple saved models
- fallback from `pipe_inference` to `pipe_train`
- fallback feature-column discovery from the scaler
- default threshold behavior
- missing-column and length validation
- real saved-model loading from `models/`

## Working Conventions

- Prefer extending reusable logic in `utils/` instead of copying notebook code.
- Update `utils/feature_columns.json` when introducing new curated feature groups.
- Keep patient-identifiable data outside the repository.
- Treat the saved-model inference path as a stable surface; changes there should be covered by tests and reflected in the API contract.
