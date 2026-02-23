# Respiratory Diagnostics Toolkit

Utilities and notebooks to clean the EFRAIM/BAZEA clinical registries, engineer reproducible features, and benchmark machine-learning or rule-based respiratory diagnostic scores. Everything lives in `utils/` for scripted reuse, while notebooks under the project root capture exploratory work.

## Repository Map
- `analyse_data/` – intermediate CSVs plus exploratory notebooks (e.g., `efraim_analyse.ipynb`).
- `utils/` – reusable Python modules for feature engineering, modeling, interpretability, and data QA.
- `dev_algo.ipynb`, `bazea_dataset.ipynb` – main notebooks that orchestrate full experiments.
- `requirements.txt` – baseline dependencies for data handling, plotting, and scikit-learn pipelines.
- `get_rpps.py` – helper to look up physicians in the official RPPS registry (requires local TXT export).

## Quick Start
1. **Python** – The bundled virtual environment targets Python 3.13. Create/activate your own if needed:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. **Dependencies** – Install the core stack:
   ```powershell
   pip install -r requirements.txt
   ```
   Some advanced scripts expect extra packages that are not pinned in `requirements.txt` (install on demand):
   ```powershell
   pip install shap xgboost lightgbm catboost scikit-multilearn tabpfn
   ```
3. **Data drops** – Place the cleaned EFRAIM/BAZEA exports (CSV or XLSX) inside `analyse_data/`. Large source files (CRF PDFs, raw EDC dumps) stay outside Git.
4. **RPPS registry (optional)** – Update `DATA_FILE_PATH` in `get_rpps.py` if your `ps-libreacces-savoirfaire.txt` lives elsewhere.

## Typical Workflow
1. **Explore quality** – Load the raw dataframe inside a notebook or script and call the helpers:
   ```python
   import pandas as pd
   from utils import data_quality, stats_dataset

   raw = pd.read_excel("analyse_data/efraim_export.xlsx")
   data_quality.analyser_nan(raw)
   stats_dataset.analyser_variables_binaires(raw[diagnosis_cols])
   ```
2. **Select & clean features** – Start from the curated JSON lists:
   ```python
   from utils.feature_loader import load_columns
   from utils.feature_transformer import transform_features

   selected = load_columns("utils/feature_columns.json", sections=["clinical_core"])
   df = raw[selected]
   features = transform_features(df)
   ```
   `transform_features` handles binarization, scaling, medical heuristics (HSCT/GvHD, neutropenia, radiology groupings, etc.), and gracefully drops missing inputs.
3. **Prepare targets** – Align binary etiologies or any multilabel objective:
   ```python
   y = raw[[c for c in raw.columns if c.startswith("Etiology_")]]
   ```
   Utilities such as `stats_dataset.fusionner_labels` merge granular outcomes into broader classes when needed.
4. **Train / evaluate** – Chain the modeling helpers:
   ```python
   from utils.algo_prediction import (
       preparer_jeu_xy,
       normaliser_features,
       entrainer_modele_multilabel,
       evaluer_modele_multilabel,
   )
   from utils.models_and_metrics import get_models

   X_train, X_test, y_train, y_test, labels = preparer_jeu_xy(features, y)
   X_train_sc, X_test_sc, scaler, cols = normaliser_features(X_train, X_test)
   model = get_models(y_train)["Random Forest"]
   clf = entrainer_modele_multilabel(X_train_sc, y_train, model)
   metrics = evaluer_modele_multilabel(clf, X_test_sc, y_test)
   ```
   Switch to `compare_models_metric` for automated leaderboard-style comparisons, or use `train_and_optimize_threshold_PR` for binary problems.
5. **Interpret & stress-test** – Run `utils/feature_importance.py` for native/permutation importances, SHAP summaries, and `utils/analyse_sensibilite.py` for partial dependence analysis. Visualization helpers live in `utils/visualisation.py`.
6. **Document experiments** – Notebooks (`dev_algo.ipynb`, `bazea_dataset.ipynb`) already import these utilities; open them with JupyterLab and keep raw outputs (plots, CSVs) inside `analyse_data/`.

## Using the RPPS Helper
```powershell
python get_rpps.py  # edit the script or import verifier_rpps() in your own notebook
```
`verifier_rpps` caches the big TXT file, so repeated queries stay fast. Pass an override path if the registry dump moves.

## Tips
- Keep personally identifiable information (PII) outside the repository and feed only anonymized tables into the notebooks.
- When adding new features, update `utils/feature_columns.json` and extend `transform_features` instead of scattering logic across notebooks.
- Prefer calling `utils` modules from notebooks/scripts rather than copy/pasting code—this keeps experiment history reproducible.

