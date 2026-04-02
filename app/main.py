from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.services.predictor import (
        build_prediction_summary,
        predict_orders_dataframe,
        save_prediction_results,
    )
    from app.services.storage import upload_file_to_blob
except ModuleNotFoundError:
    # Azure App Service can start from inside the app folder when we use --chdir app.
    from services.predictor import (  # type: ignore
        build_prediction_summary,
        predict_orders_dataframe,
        save_prediction_results,
    )
    from services.storage import upload_file_to_blob  # type: ignore


app = FastAPI(
    title="Supply Chain Fraud Detection API",
    description="Upload ERP order CSV data and detect suspicious transactions.",
    version="1.0.0",
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


@app.post("/predict")
async def predict_from_csv(file: UploadFile = File(...)) -> dict:
    """
    Accept a CSV file upload, run predictions, save the result locally,
    and return summary data plus top suspicious rows.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name was provided.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        file_bytes = await file.read()
        dataframe = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded CSV file: {error}",
        ) from error

    try:
        result_dataframe = predict_orders_dataframe(dataframe)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = Path(file.filename).stem
    output_filename = f"{safe_filename}_predictions_{timestamp}.csv"
    output_path = save_prediction_results(result_dataframe, output_filename)
    azure_upload_result = upload_file_to_blob(
        local_file_path=output_path,
        container_name="outputs",
        blob_name=output_filename,
    )

    summary = build_prediction_summary(result_dataframe)
    suspicious_rows = (
        result_dataframe.sort_values(by="fraud_risk_score", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    response = {
        "message": "Prediction completed successfully.",
        "uploaded_file_name": file.filename,
        "output_file_path": str(output_path),
        "blob_upload": azure_upload_result,
        "summary": summary,
        "top_suspicious_records": suspicious_rows,
    }
    return response
