import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from app.services.features import add_engineered_features, get_model_input_dataframe
    from app.services.preprocess import preprocess_orders_dataframe
except ModuleNotFoundError:
    from services.features import add_engineered_features, get_model_input_dataframe  # type: ignore
    from services.preprocess import preprocess_orders_dataframe  # type: ignore


# Single source of truth for model display metadata. The training script
# imports this too, so the API and the benchmark JSON always agree.
MODEL_INFO = {
    "random_forest": {
        "display_name": "Random Forest",
        "family": "Bagging ensemble",
        "supervised": True,
        "note": None,
    },
    "xgboost": {
        "display_name": "XGBoost",
        "family": "Gradient boosting",
        "supervised": True,
        "note": None,
    },
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "family": "Linear model",
        "supervised": True,
        "note": None,
    },
    "isolation_forest": {
        "display_name": "Isolation Forest",
        "family": "Anomaly detection",
        "supervised": False,
        "note": (
            "Unsupervised anomaly detector: it never sees fraud labels, so its "
            "precision/recall are expected to be lower. Its value is catching "
            "unusual orders that labeled training data may have missed."
        ),
    },
}

MODEL_BUNDLE_FILENAME = "fraud_models.pkl"
LEGACY_MODEL_FILENAME = "fraud_model.pkl"
BENCHMARK_METRICS_FILENAME = "model_metrics.json"

# Loaded once per process instead of once per request.
_bundle_cache: dict | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_model_bundle(force_reload: bool = False) -> dict:
    """
    Load the multi-model bundle saved by scripts/train_baseline.py.

    Falls back to the legacy single-model artifact (wrapped to look like
    a one-model bundle) so older model files keep working.
    """
    global _bundle_cache
    if _bundle_cache is not None and not force_reload:
        return _bundle_cache

    models_dir = _project_root() / "models"
    bundle_path = models_dir / MODEL_BUNDLE_FILENAME
    if bundle_path.exists():
        _bundle_cache = joblib.load(bundle_path)
        return _bundle_cache

    legacy_path = models_dir / LEGACY_MODEL_FILENAME
    if legacy_path.exists():
        legacy_artifact = joblib.load(legacy_path)
        _bundle_cache = {
            "models": {
                "random_forest": {
                    "model": legacy_artifact["model"],
                    "supervised": True,
                    "tuned_threshold": 0.5,
                },
            },
            "feature_columns": legacy_artifact["feature_columns"],
            "best_model": "random_forest",
        }
        return _bundle_cache

    raise FileNotFoundError(
        f"No model file found in {models_dir}. "
        "Please run scripts/train_baseline.py first."
    )


def load_model_artifact() -> dict:
    """
    Backward-compatible accessor for the legacy single-model artifact shape.
    """
    bundle = load_model_bundle()
    best_entry = bundle["models"][bundle["best_model"]]
    return {
        "model": best_entry["model"],
        "feature_columns": bundle["feature_columns"],
    }


def load_benchmark_metrics() -> dict | None:
    """
    Read the offline benchmark JSON written at training time, if present.
    """
    metrics_path = _project_root() / "models" / BENCHMARK_METRICS_FILENAME
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_model_name(model_name: str | None = None) -> tuple[str, dict]:
    """
    Validate a requested model name against the bundle.

    Returns the resolved name (defaulting to the best supervised model)
    and its bundle entry. Raises ValueError for unknown names.
    """
    bundle = load_model_bundle()
    if not model_name:
        model_name = bundle["best_model"]

    if model_name not in bundle["models"]:
        available = ", ".join(sorted(bundle["models"].keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    return model_name, bundle["models"][model_name]


def list_available_models() -> list[dict]:
    """
    Describe every model in the bundle for the /models endpoint.
    """
    bundle = load_model_bundle()
    models = []
    for name, entry in bundle["models"].items():
        info = MODEL_INFO.get(name, {})
        models.append(
            {
                "name": name,
                "display_name": info.get("display_name", name),
                "family": info.get("family", "unknown"),
                "supervised": bool(entry.get("supervised", True)),
                "tuned_threshold": entry.get("tuned_threshold"),
                "note": info.get("note"),
                "is_best": name == bundle["best_model"],
            }
        )
    return models


def _score_with_entry(entry: dict, model_input: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce (risk_scores, suspicious_flags) for one bundle entry.

    Supervised models: probability of fraud, flagged at the tuned threshold.
    Isolation Forest: normalized anomaly score, flagged by its own
    contamination-based prediction.
    """
    model = entry["model"]

    if entry.get("supervised", True):
        probabilities = model.predict_proba(model_input)[:, 1]
        threshold = entry.get("tuned_threshold") or 0.5
        flags = (probabilities >= threshold).astype(int)
        return probabilities.round(4), flags

    raw_scores = -model.decision_function(model_input)
    score_min = entry.get("score_min", float(raw_scores.min()))
    score_max = entry.get("score_max", float(raw_scores.max()))
    score_range = max(score_max - score_min, 1e-9)
    risk_scores = np.clip((raw_scores - score_min) / score_range, 0.0, 1.0)
    flags = (model.predict(model_input) == -1).astype(int)
    return risk_scores.round(4), flags


def predict_orders_dataframe(
    dataframe: pd.DataFrame,
    model_name: str | None = None,
) -> pd.DataFrame:
    """
    Predict suspicious orders with one model (default: best by PR-AUC).

    Output columns added:
    - fraud_risk_score
    - suspicious_flag
    """
    cleaned_dataframe = preprocess_orders_dataframe(dataframe)
    featured_dataframe = add_engineered_features(cleaned_dataframe)

    resolved_name, entry = resolve_model_name(model_name)
    model_input = get_model_input_dataframe(featured_dataframe)

    risk_scores, flags = _score_with_entry(entry, model_input)

    result_dataframe = featured_dataframe.copy()
    result_dataframe["fraud_risk_score"] = risk_scores
    result_dataframe["suspicious_flag"] = flags
    return result_dataframe


def compare_orders_dataframe(dataframe: pd.DataFrame) -> dict:
    """
    Score the data with every model in the bundle.

    Returns the comparison DataFrame (per-model scores and flags plus a
    consensus count), per-model summaries, consensus statistics, and
    live metrics when the file contains a fraud_label column.
    """
    cleaned_dataframe = preprocess_orders_dataframe(dataframe)
    featured_dataframe = add_engineered_features(cleaned_dataframe)

    bundle = load_model_bundle()
    model_input = get_model_input_dataframe(featured_dataframe)

    result_dataframe = featured_dataframe.copy()
    model_names = list(bundle["models"].keys())
    supervised_score_columns = []
    flag_columns = []
    scores_by_model: dict[str, np.ndarray] = {}
    flags_by_model: dict[str, np.ndarray] = {}

    for name in model_names:
        entry = bundle["models"][name]
        risk_scores, flags = _score_with_entry(entry, model_input)
        scores_by_model[name] = risk_scores
        flags_by_model[name] = flags

        result_dataframe[f"{name}_risk_score"] = risk_scores
        result_dataframe[f"{name}_suspicious_flag"] = flags
        flag_columns.append(f"{name}_suspicious_flag")
        if entry.get("supervised", True):
            supervised_score_columns.append(f"{name}_risk_score")

    result_dataframe["avg_risk_score"] = (
        result_dataframe[supervised_score_columns].mean(axis=1).round(4)
    )
    result_dataframe["consensus_count"] = result_dataframe[flag_columns].sum(axis=1)

    # Primary verdict columns mirror the best model, so the comparison CSV
    # stays compatible with the single-model output format.
    best_model = bundle["best_model"]
    result_dataframe["fraud_risk_score"] = scores_by_model[best_model]
    result_dataframe["suspicious_flag"] = flags_by_model[best_model]

    total_models = len(model_names)
    majority_needed = (total_models // 2) + 1
    consensus_counts = result_dataframe["consensus_count"]
    consensus = {
        "models_total": total_models,
        "majority_needed": majority_needed,
        "rows_flagged_by_any": int((consensus_counts >= 1).sum()),
        "rows_flagged_by_majority": int((consensus_counts >= majority_needed).sum()),
        "rows_flagged_by_all": int((consensus_counts == total_models).sum()),
    }

    per_model_summary = {}
    for name in model_names:
        entry = bundle["models"][name]
        info = MODEL_INFO.get(name, {})
        flags = flags_by_model[name]
        scores = scores_by_model[name]
        flagged_scores = scores[flags == 1]
        per_model_summary[name] = {
            "display_name": info.get("display_name", name),
            "supervised": bool(entry.get("supervised", True)),
            "tuned_threshold": entry.get("tuned_threshold"),
            "suspicious_rows": int(flags.sum()),
            "average_flagged_risk_score": (
                round(float(flagged_scores.mean()), 4) if len(flagged_scores) else 0.0
            ),
        }

    live_metrics = None
    if "fraud_label" in result_dataframe.columns:
        live_metrics = _build_live_metrics(
            result_dataframe["fraud_label"].astype(int),
            scores_by_model,
            flags_by_model,
        )

    return {
        "result_dataframe": result_dataframe,
        "models_compared": model_names,
        "best_model": best_model,
        "per_model_summary": per_model_summary,
        "consensus": consensus,
        "live_metrics": live_metrics,
    }


def _build_live_metrics(
    labels: pd.Series,
    scores_by_model: dict,
    flags_by_model: dict,
) -> dict:
    """
    Compute metrics against the labels found in the uploaded file.

    Only possible when the upload includes fraud_label; ranking metrics
    additionally need both classes to be present.
    """
    has_both_classes = labels.nunique() > 1

    per_model = {}
    for name, flags in flags_by_model.items():
        entry = {
            "accuracy": round(float(accuracy_score(labels, flags)), 4),
            "precision": round(float(precision_score(labels, flags, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, flags, zero_division=0)), 4),
            "f1_score": round(float(f1_score(labels, flags, zero_division=0)), 4),
            "roc_auc": None,
            "pr_auc": None,
        }
        if has_both_classes:
            scores = scores_by_model[name]
            entry["roc_auc"] = round(float(roc_auc_score(labels, scores)), 4)
            entry["pr_auc"] = round(float(average_precision_score(labels, scores)), 4)
        per_model[name] = entry

    return {
        "label_rows": int(len(labels)),
        "fraud_rows_in_file": int(labels.sum()),
        "both_classes_present": bool(has_both_classes),
        "note": (
            "Metrics computed against the fraud_label column found in the "
            "uploaded file. If this file overlaps with training data, these "
            "numbers will look optimistic."
        ),
        "per_model": per_model,
    }


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
    output_path = _project_root() / "outputs" / output_filename
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
