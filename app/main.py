from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.services.predictor import (
        MODEL_INFO,
        build_prediction_summary,
        compare_orders_dataframe,
        list_available_models,
        load_benchmark_metrics,
        load_model_bundle,
        predict_orders_dataframe,
        resolve_model_name,
        save_prediction_results,
    )
    from app.services.storage import upload_file_to_blob
except ModuleNotFoundError:
    # Azure App Service can start from inside the app folder when we use --chdir app.
    from services.predictor import (  # type: ignore
        MODEL_INFO,
        build_prediction_summary,
        compare_orders_dataframe,
        list_available_models,
        load_benchmark_metrics,
        load_model_bundle,
        predict_orders_dataframe,
        resolve_model_name,
        save_prediction_results,
    )
    from services.storage import upload_file_to_blob  # type: ignore


app = FastAPI(
    title="Supply Chain Fraud Detection API",
    description=(
        "Upload ERP order CSV data and detect suspicious transactions. "
        "Supports single-model prediction and multi-model comparison."
    ),
    version="2.0.0",
)


# This allows our future frontend to call the backend from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    """
    Simple endpoint to confirm that the API is running.
    """
    return {
        "status": "ok",
        "message": "Fraud detection API is running.",
    }


@app.get("/models")
def get_models() -> dict:
    """
    List the trained models and the offline benchmark metrics
    produced by scripts/train_baseline.py.
    """
    try:
        models = list_available_models()
        bundle = load_model_bundle()
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "models": models,
        "best_model": bundle["best_model"],
        "benchmark": load_benchmark_metrics(),
    }


async def _read_upload_to_dataframe(file: UploadFile) -> pd.DataFrame:
    """
    Shared upload validation and CSV parsing for the prediction endpoints.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name was provided.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        file_bytes = await file.read()
        return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded CSV file: {error}",
        ) from error


def _save_and_upload(result_dataframe: pd.DataFrame, upload_name: str, suffix: str) -> tuple:
    """
    Persist the scored CSV locally and attempt the optional Blob upload.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = Path(upload_name).stem
    output_filename = f"{safe_filename}_{suffix}_{timestamp}.csv"
    output_path = save_prediction_results(result_dataframe, output_filename)
    azure_upload_result = upload_file_to_blob(
        local_file_path=output_path,
        container_name="outputs",
        blob_name=output_filename,
    )
    return output_path, azure_upload_result


@app.post("/predict")
async def predict_from_csv(
    file: UploadFile = File(...),
    model: str | None = None,
) -> dict:
    """
    Accept a CSV file upload, run predictions with one model
    (default: the best model from training), save the result locally,
    and return summary data plus top suspicious rows.
    """
    dataframe = await _read_upload_to_dataframe(file)

    try:
        model_name, model_entry = resolve_model_name(model)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        result_dataframe = predict_orders_dataframe(dataframe, model_name)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    output_path, azure_upload_result = _save_and_upload(
        result_dataframe, file.filename, "predictions"
    )

    summary = build_prediction_summary(result_dataframe)
    suspicious_rows = (
        result_dataframe.sort_values(by="fraud_risk_score", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    model_info = MODEL_INFO.get(model_name, {})
    response = {
        "message": "Prediction completed successfully.",
        "uploaded_file_name": file.filename,
        "model_used": model_name,
        "model_display_name": model_info.get("display_name", model_name),
        "threshold_used": model_entry.get("tuned_threshold"),
        "output_file_path": str(output_path),
        "blob_upload": azure_upload_result,
        "summary": summary,
        "top_suspicious_records": suspicious_rows,
    }
    return response


@app.post("/compare")
async def compare_from_csv(file: UploadFile = File(...)) -> dict:
    """
    Accept a CSV file upload and score it with every trained model.

    Returns per-model summaries, consensus statistics, the offline
    benchmark, and live metrics when the file contains fraud_label.
    """
    dataframe = await _read_upload_to_dataframe(file)

    try:
        comparison = compare_orders_dataframe(dataframe)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {error}",
        ) from error

    result_dataframe = comparison["result_dataframe"]
    output_path, azure_upload_result = _save_and_upload(
        result_dataframe, file.filename, "comparison"
    )

    summary = build_prediction_summary(result_dataframe)
    top_records = (
        result_dataframe.sort_values(
            by=["avg_risk_score", "consensus_count"], ascending=False
        )
        .head(10)
        .to_dict(orient="records")
    )

    response = {
        "message": "Model comparison completed successfully.",
        "uploaded_file_name": file.filename,
        "models_compared": comparison["models_compared"],
        "best_model": comparison["best_model"],
        "output_file_path": str(output_path),
        "blob_upload": azure_upload_result,
        "summary": summary,
        "per_model_summary": comparison["per_model_summary"],
        "consensus": comparison["consensus"],
        "live_metrics": comparison["live_metrics"],
        "benchmark": load_benchmark_metrics(),
        "top_suspicious_records": top_records,
    }
    return response
