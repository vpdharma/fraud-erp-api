from pathlib import Path

import joblib
import pandas as pd

try:
    from app.services.features import add_engineered_features, get_model_input_dataframe
    from app.services.preprocess import preprocess_orders_dataframe
except ModuleNotFoundError:
    from services.features import add_engineered_features, get_model_input_dataframe  # type: ignore
    from services.preprocess import preprocess_orders_dataframe  # type: ignore


def load_model_artifact() -> dict:
    """
    Load the saved model file from the models folder.

    The artifact contains:
    - the trained model
    - the list of feature column names
    """
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "fraud_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please run scripts/train_baseline.py first."
        )

    model_artifact = joblib.load(model_path)
    return model_artifact


def predict_orders_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Predict suspicious orders from a pandas DataFrame.

    Output columns added:
    - fraud_risk_score
    - suspicious_flag
    """
    cleaned_dataframe = preprocess_orders_dataframe(dataframe)
    featured_dataframe = add_engineered_features(cleaned_dataframe)

    model_artifact = load_model_artifact()
    model = model_artifact["model"]

    model_input = get_model_input_dataframe(featured_dataframe)

    prediction_probabilities = model.predict_proba(model_input)[:, 1]
    prediction_labels = model.predict(model_input)

    result_dataframe = featured_dataframe.copy()
    result_dataframe["fraud_risk_score"] = prediction_probabilities.round(4)
    result_dataframe["suspicious_flag"] = prediction_labels.astype(int)

    return result_dataframe


def predict_orders_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file and return prediction results.
    """
    dataframe = pd.read_csv(csv_path)
    return predict_orders_dataframe(dataframe)


def save_prediction_results(
    dataframe: pd.DataFrame,
    output_filename: str = "prediction_results.csv",
) -> Path:
    """
    Save prediction results in the outputs folder.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "outputs" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def build_prediction_summary(dataframe: pd.DataFrame) -> dict:
    """
    Build a simple response summary for the frontend or API.
    """
    suspicious_dataframe = dataframe[dataframe["suspicious_flag"] == 1].copy()

    if suspicious_dataframe.empty:
        average_risk_score = 0.0
    else:
        average_risk_score = float(suspicious_dataframe["fraud_risk_score"].mean())

    summary = {
        "total_rows": int(len(dataframe)),
        "suspicious_rows": int(len(suspicious_dataframe)),
        "average_suspicious_risk_score": round(average_risk_score, 4),
    }
    return summary
