# Fraud and Abnormal Order Detection in Supply Chain ERP Transactions

## Project purpose

This project is a beginner-friendly final-year B.Tech IT major project.

It detects suspicious ERP-style supply chain orders from CSV data by:

- reading uploaded order data
- cleaning and preparing the data
- creating fraud-related business features
- using a machine learning model to score each order
- showing suspicious records on a web page
- optionally saving results to Azure Blob Storage

## What this project does

The system accepts a CSV file containing supply chain order data such as:

- order amount
- quantity
- discount percent
- billing and shipping state
- customer age and order history
- refund amount
- return count

Then it predicts:

- `fraud_risk_score`
- `suspicious_flag`

`fraud_risk_score` is a numeric risk score between 0 and 1.

`suspicious_flag` is:

- `1` for suspicious
- `0` for not suspicious

## Beginner-friendly word meanings

### What is a CSV?

A CSV is a simple file used to store table data.

You can open it in:

- Excel
- Google Sheets
- Notepad

### What is a virtual environment?

A virtual environment is a private Python workspace for one project.

It helps avoid conflicts between project libraries.

### What does `pip install` do?

`pip install` downloads and installs Python libraries listed in `requirements.txt`.

### What is an API?

An API is a way for one software program to talk to another software program.

In this project, the website sends a CSV file to the FastAPI backend through an API.

### What is FastAPI?

FastAPI is a Python framework used to build backend APIs quickly and clearly.

### What is a model file `.pkl`?

A `.pkl` file is a saved Python object.

In this project, `models/fraud_model.pkl` stores the trained machine learning model so we can reuse it later without training again.

### What is Blob Storage?

Azure Blob Storage is Azure's cloud file storage service.

We use it to store uploaded or generated output files in the cloud.

### What is an environment variable?

An environment variable is a setting stored outside the code.

Example:

- `AZURE_STORAGE_CONNECTION_STRING`

This is useful for secrets because we do not hard-code them in Python files.

### What does deployment mean?

Deployment means moving your project from your own computer to a live cloud environment so other people can access it from the internet.

## Project folder structure

```text
fraud_supply_chain_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── services/
│       ├── __init__.py
│       ├── preprocess.py
│       ├── features.py
│       ├── predictor.py
│       └── storage.py
│
├── scripts/
│   ├── generate_sample_data.py
│   └── train_baseline.py
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

## Which file does what?

- `scripts/generate_sample_data.py`
  creates synthetic ERP-like order data

- `app/services/preprocess.py`
  cleans data, standardizes columns, removes duplicates, fills missing values

- `app/services/features.py`
  creates business fraud features such as shipping mismatch and return ratio

- `scripts/train_baseline.py`
  trains the baseline Random Forest model and saves it

- `app/services/predictor.py`
  loads the saved model and predicts suspicious orders

- `app/main.py`
  creates the FastAPI backend

- `app/services/storage.py`
  uploads files to Azure Blob Storage if Azure is configured

- `frontend/index.html`
  website structure

- `frontend/styles.css`
  website design

- `frontend/app.js`
  website logic for upload and showing results

## Local setup on Windows

Open PowerShell and run:

```powershell
cd E:\Major_Project\fraud_supply_chain_project
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

If activation is blocked in PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate
```

## How to generate sample data

Run:

```powershell
python scripts\generate_sample_data.py
```

Expected output:

- `data/raw/orders.csv` is created
- console shows total rows and suspicious row count

## How to train the model

Run:

```powershell
python scripts\train_baseline.py
```

Expected output:

- confusion matrix
- classification report
- ROC-AUC
- `models/fraud_model.pkl` created

## How to run the backend

Run:

```powershell
uvicorn app.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## How to run the frontend

Open a second PowerShell window and run:

```powershell
cd E:\Major_Project\fraud_supply_chain_project\frontend
E:\Major_Project\fraud_supply_chain_project\.venv\Scripts\python.exe -m http.server 5500
```

Then open:

- `http://127.0.0.1:5500`

Keep the backend URL as:

- `http://127.0.0.1:8000`

Upload:

- `data/raw/orders.csv`

## Optional Azure Blob Storage usage

Set Azure Storage connection string in PowerShell:

```powershell
$env:AZURE_STORAGE_CONNECTION_STRING="PASTE_YOUR_CONNECTION_STRING_HERE"
```

Then run the backend:

```powershell
uvicorn app.main:app --reload
```

If Azure is configured correctly, prediction outputs can also upload to the Azure Blob container.

To disable Azure upload in the current PowerShell window:

```powershell
Remove-Item Env:AZURE_STORAGE_CONNECTION_STRING
```

## Azure deployment notes

Recommended student-friendly services:

- frontend: Azure Static Web Apps
- backend: Azure App Service
- file storage: Azure Blob Storage
- monitoring: Application Insights

Recommended cheap settings:

- Storage Account: Standard, LRS
- App Service Plan: Basic B1 or Free F1 if available and acceptable
- Static Web Apps: Free plan for basic demo usage

For FastAPI on Azure App Service, use a startup command similar to:

```text
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
```

## Future improvements

Later versions can add:

- XGBoost
- Isolation Forest
- Autoencoder
- alert workflow
- Application Insights monitoring in code
- better threshold tuning
- richer frontend charts

## Viva explanation summary

You can explain this project in simple steps:

1. CSV order data is uploaded.
2. The backend cleans the data.
3. Fraud-related business features are created.
4. A Random Forest model predicts suspicious transactions.
5. The website shows suspicious rows and risk score.
6. Results are saved locally and optionally in Azure Blob Storage.
