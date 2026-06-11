# Project Handoff

## 1. What This Project Does And Its Current State

This project is a final-year B.Tech IT system for detecting fraud and abnormal orders in supply chain ERP-style transaction data. It takes a CSV upload, cleans and standardizes the data, engineers fraud-related business features, runs a trained Random Forest model, and returns a `fraud_risk_score` plus a binary `suspicious_flag`. The backend can also save prediction outputs locally and optionally upload the generated CSV to Azure Blob Storage.

The current repo is in a mostly working state:

- The synthetic dataset generator works.
- The preprocessing, feature engineering, prediction, and output-saving pipeline works.
- The FastAPI backend exposes `/health` and `/predict`.
- The static frontend can upload a CSV and display results.
- Azure Blob upload is implemented as an optional backend-only step.
- The trained model artifact already exists at `models/fraud_model.pkl`.
- Generated sample data and example prediction outputs already exist in the repo workspace.
- The project also now includes LaTeX report/presentation files in the repo root.

## 2. Architecture And Key Design Decisions

### High-level architecture

- `scripts/generate_sample_data.py` creates synthetic ERP-like order data with a built-in fraud label.
- `scripts/train_baseline.py` preprocesses that data, adds engineered features, trains a Random Forest model, evaluates it, and saves the model artifact.
- `app/main.py` exposes the HTTP API.
- `app/services/preprocess.py` cleans incoming transaction data.
- `app/services/features.py` creates fraud-oriented business features and defines the model input columns.
- `app/services/predictor.py` loads the saved model, predicts, and writes output CSVs.
- `app/services/storage.py` uploads the generated output file to Azure Blob Storage if Azure is configured.
- `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css` form the browser UI.

### Why it was built this way

- FastAPI was chosen because it is lightweight, easy to document, and simple to wire into a machine-learning inference endpoint.
- A static frontend served via `python -m http.server` keeps the UI simple and avoids adding a heavy frontend framework just to upload files and display results.
- Random Forest is used as the baseline model because the problem is tabular, the features are structured, and the model is easy to explain and deploy.
- Business-style engineered features were added because raw ERP fields alone are not always enough to expose suspicious behavior clearly.
- Azure Blob upload is optional by design so the project works locally even when cloud configuration is absent.
- Azure credentials are read from backend environment variables instead of frontend code so secrets are not exposed in the browser.
- The code includes import fallbacks so Azure App Service can start either from the repo root or from inside `app/` depending on deployment startup behavior.

## 3. What Is Complete Vs. In-Progress

### Complete

- Synthetic data generation.
- Data preprocessing and schema validation.
- Fraud feature engineering.
- Baseline model training and serialization.
- Backend API.
- Frontend upload and display flow.
- Local prediction output saving.
- Optional Azure Blob upload.
- Report and presentation LaTeX artifacts.

### In-progress

- Improving recall and overall fraud detection quality.
- Threshold tuning and class-imbalance handling.
- Comparing the baseline with stronger alternatives like XGBoost or anomaly-detection methods.
- Better operational feedback around Azure upload success/failure.
- More robust monitoring/logging for a real deployment.
- Testing on more realistic data than the current synthetic dataset.

## 4. The Exact Next Task I Was Working On

The last active task in the repo was polishing the presentation materials, especially:

- tightening the methodology slide layout so the diagram fit cleanly in the frame,
- and preparing slide-by-slide speaker notes / presentation script.

If you mean the next engineering task after that presentation work, the most important open item is improving model recall through threshold tuning and imbalance handling.

## 5. Known Bugs, Gotchas, And Intentional Oddities

- Blob upload is optional. If `AZURE_STORAGE_CONNECTION_STRING` is not set, the backend will still run and simply return a local output path.
- The backend uploads the generated prediction CSV, not the original CSV the user submitted. That is intentional.
- The frontend shows either the Blob URL or the local output path. If Azure upload fails, it falls back to the local path.
- `frontend/index.html` defaults the API URL to `http://127.0.0.1:8000`. That is correct for local development, but it must be changed when pointing at a deployed backend.
- `app/main.py` has a fallback import path for Azure App Service startup quirks. That looks odd at first glance but is intentional.
- The repo does not currently have an automated test suite.
- The synthetic dataset is deliberately constructed, so model metrics are good for pipeline validation but not proof of real-world fraud performance.
- The current baseline has strong ranking quality but weak recall at the default threshold. That is a modeling limitation, not a broken pipeline.
- The generated outputs in `outputs/` are ignored by git, so they will not be tracked unless explicitly copied elsewhere.

## 6. Build, Run, And Test Commands

### Local setup

```powershell
cd E:\Major_Project\fraud_supply_chain_project
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Generate sample data

```powershell
python scripts\generate_sample_data.py
```

### Train the baseline model

```powershell
python scripts\train_baseline.py
```

### Run the backend

```powershell
uvicorn app.main:app --reload
```

Useful backend URLs:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### Run the frontend

```powershell
cd E:\Major_Project\fraud_supply_chain_project\frontend
E:\Major_Project\fraud_supply_chain_project\.venv\Scripts\python.exe -m http.server 5500
```

Frontend URL:

- `http://127.0.0.1:5500`

### Manual smoke tests

- Open `/health` and confirm the backend is alive.
- Upload `data/raw/orders.csv` through the frontend.
- Confirm the response includes `summary`, `top_suspicious_records`, and `blob_upload`.
- Verify a new CSV appears in `outputs/`.
- If Azure is configured, verify the file also appears in the `outputs` Blob container.

### Azure startup command

```text
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
```

## 7. Environment And Config Setup

- The project uses a Python virtual environment at `.venv/`.
- Azure Blob Storage is configured with the backend environment variable `AZURE_STORAGE_CONNECTION_STRING`.
- In local development, set that variable in PowerShell before starting `uvicorn` if you want cloud upload enabled.
- In Azure App Service, set the same key under App Settings, not in the frontend.
- There is no committed `.env` file in the current repo workflow.
- The current `.gitignore` already excludes:
  - `.venv/`
  - `__pycache__/`
  - `*.pyc`, `*.pyo`, `*.pyd`
  - `outputs/*.csv`
  - `data/processed/*.csv`
  - `.azure/`
  - `.azure-local/`
  - `.vscode/`
  - `.idea/`
  - `backend_appservice*.zip`
- If you choose to introduce a local `.env` file later, keep it untracked unless you also update `.gitignore`.

## 8. Dead Ends Already Ruled Out

- Do not move the Blob credential into frontend JavaScript. That is insecure and unnecessary.
- Do not replace the backend with a database-backed workflow just to get the current project working. The file-upload batch pipeline is the intended design.
- Do not treat the optional Blob upload as mandatory. The app is supposed to keep working locally even when Azure is not configured.
- Do not look for a hidden alternate storage provider such as S3 or Supabase. Only Azure Blob Storage is implemented.
- Do not spend time chasing a missing test suite. There is no automated test harness yet, so validation is currently manual.
- Do not expect the backend to upload the input CSV. It uploads the generated prediction result after `/predict` finishes.
- Do not assume the frontend needs a framework rewrite. The static HTML/CSS/JS approach is intentional and sufficient for the current scope.

## Useful Files To Know

- `app/main.py`
- `app/services/preprocess.py`
- `app/services/features.py`
- `app/services/predictor.py`
- `app/services/storage.py`
- `scripts/generate_sample_data.py`
- `scripts/train_baseline.py`
- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`
- `project_progress_report.tex`
- `project_progress_presentation.tex`
- `myref.bib`
