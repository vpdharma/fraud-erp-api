import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.services.features import (
    MODEL_FEATURE_COLUMNS,
    add_engineered_features,
    save_featured_dataframe,
    split_features_and_target,
)
from app.services.predictor import MODEL_INFO
from app.services.preprocess import (
    load_and_preprocess_csv,
    save_processed_dataframe,
)


def build_supervised_models(y_train) -> dict:
    """
    Create the supervised models we want to compare.

    Each model family is configured for the ~10% fraud class imbalance:
    - Random Forest and Logistic Regression use class_weight="balanced"
    - XGBoost uses scale_pos_weight = negatives / positives
    - Logistic Regression gets a StandardScaler because linear models
      are sensitive to feature scale (tree models are not)
    """
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def tune_decision_threshold(model, X_train, y_train) -> float:
    """
    Pick a decision threshold from out-of-fold predictions on the
    training data, so the test set is never used for tuning.

    We take the threshold that maximizes F1 on the precision-recall
    curve. This is the fix for the "high AUC but tiny recall at 0.5"
    problem of the original baseline.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probabilities = cross_val_predict(
        clone(model), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, oof_probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_scores = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    if len(thresholds) == 0 or float(f1_scores.max()) <= 0.0:
        return 0.5

    best_threshold = float(thresholds[int(np.argmax(f1_scores))])
    return round(best_threshold, 4)


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def confusion_matrix_as_list(y_true, y_pred) -> list:
    return [[int(cell) for cell in row] for row in confusion_matrix(y_true, y_pred)]


def evaluate_supervised_model(model, threshold, X_test, y_test) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    default_predictions = model.predict(X_test)
    tuned_predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "default_threshold": 0.5,
        "tuned_threshold": threshold,
        "metrics_default": classification_metrics(y_test, default_predictions),
        "metrics_tuned": classification_metrics(y_test, tuned_predictions),
        "confusion_matrix_default": confusion_matrix_as_list(y_test, default_predictions),
        "confusion_matrix_tuned": confusion_matrix_as_list(y_test, tuned_predictions),
    }


def train_isolation_forest(X_train, y_train, X_test, y_test) -> tuple[dict, dict]:
    """
    Train the unsupervised anomaly detector.

    Isolation Forest never sees fraud_label. Its raw anomaly scores are
    min-max normalized (using training-data bounds) into a 0-1 risk
    score so it can sit in the same comparison as the supervised models.
    """
    contamination = round(float((y_train == 1).mean()), 4)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)

    train_scores = -model.decision_function(X_train)
    score_min = float(train_scores.min())
    score_max = float(train_scores.max())
    score_range = max(score_max - score_min, 1e-9)

    test_scores = -model.decision_function(X_test)
    test_risk = np.clip((test_scores - score_min) / score_range, 0.0, 1.0)
    test_flags = (model.predict(X_test) == -1).astype(int)

    entry = {
        "model": model,
        "supervised": False,
        "score_min": score_min,
        "score_max": score_max,
        "tuned_threshold": None,
    }

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, test_risk)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_risk)), 4),
        "default_threshold": None,
        "tuned_threshold": None,
        "metrics_default": classification_metrics(y_test, test_flags),
        "metrics_tuned": None,
        "confusion_matrix_default": confusion_matrix_as_list(y_test, test_flags),
        "confusion_matrix_tuned": None,
    }
    return entry, metrics


def train_all_models() -> Path:
    """
    Train and compare all fraud detection models.

    Steps:
    1. Load raw CSV, clean it, create fraud features
    2. Stratified 80/20 train/test split (same split for every model)
    3. Tune each supervised model's threshold on out-of-fold
       training predictions, then fit on the full training set
    4. Train the Isolation Forest anomaly detector
    5. Evaluate everything on the same held-out test set
    6. Save the model bundle, the benchmark metrics JSON, and the
       legacy single-model artifact
    """
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "orders.csv"

    cleaned_dataframe = load_and_preprocess_csv(raw_csv_path)
    featured_dataframe = add_engineered_features(cleaned_dataframe)

    cleaned_output_path = save_processed_dataframe(cleaned_dataframe)
    featured_output_path = save_featured_dataframe(featured_dataframe)

    features, target = split_features_and_target(featured_dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    print("Training data prepared.")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print(f"Fraud rate in training data: {float((y_train == 1).mean()):.4f}")
    print(f"Cleaned data saved to: {cleaned_output_path}")
    print(f"Feature-engineered data saved to: {featured_output_path}\n")

    bundle_models: dict = {}
    metrics_by_model: dict = {}

    for model_name, model in build_supervised_models(y_train).items():
        display_name = MODEL_INFO[model_name]["display_name"]
        print(f"Training {display_name}...")

        threshold = tune_decision_threshold(model, X_train, y_train)
        model.fit(X_train, y_train)
        evaluation = evaluate_supervised_model(model, threshold, X_test, y_test)

        bundle_models[model_name] = {
            "model": model,
            "supervised": True,
            "tuned_threshold": threshold,
        }
        metrics_by_model[model_name] = evaluation

        tuned = evaluation["metrics_tuned"]
        print(
            f"  PR-AUC {evaluation['pr_auc']:.4f} | ROC-AUC {evaluation['roc_auc']:.4f} | "
            f"tuned threshold {threshold:.4f} -> "
            f"precision {tuned['precision']:.4f}, recall {tuned['recall']:.4f}, "
            f"F1 {tuned['f1_score']:.4f}"
        )

    print("Training Isolation Forest (unsupervised)...")
    iso_entry, iso_metrics = train_isolation_forest(X_train, y_train, X_test, y_test)
    bundle_models["isolation_forest"] = iso_entry
    metrics_by_model["isolation_forest"] = iso_metrics
    default = iso_metrics["metrics_default"]
    print(
        f"  PR-AUC {iso_metrics['pr_auc']:.4f} | ROC-AUC {iso_metrics['roc_auc']:.4f} | "
        f"contamination flags -> precision {default['precision']:.4f}, "
        f"recall {default['recall']:.4f}\n"
    )

    supervised_names = [name for name, entry in bundle_models.items() if entry["supervised"]]
    best_model = max(supervised_names, key=lambda name: metrics_by_model[name]["pr_auc"])
    print(f"Best supervised model by PR-AUC: {MODEL_INFO[best_model]['display_name']}\n")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "models": bundle_models,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "best_model": best_model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    bundle_path = models_dir / "fraud_models.pkl"
    joblib.dump(bundle, bundle_path)
    print(f"Model bundle saved to: {bundle_path}")

    # Legacy single-model artifact, kept so older code keeps working.
    legacy_artifact = {
        "model": bundle_models["random_forest"]["model"],
        "feature_columns": MODEL_FEATURE_COLUMNS,
    }
    legacy_path = models_dir / "fraud_model.pkl"
    joblib.dump(legacy_artifact, legacy_path)
    print(f"Legacy single-model artifact saved to: {legacy_path}")

    benchmark = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline_metric": "pr_auc",
        "headline_note": (
            "PR-AUC (average precision) is the headline metric because the data "
            "is imbalanced (~10% fraud); accuracy is inflated by the majority class."
        ),
        "dataset": {
            "total_rows": int(len(featured_dataframe)),
            "fraud_rows": int(target.sum()),
            "fraud_rate": round(float(target.mean()), 4),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "source": "synthetic ERP order data (data/raw/orders.csv)",
        },
        "best_model": best_model,
        "models": {
            name: {
                "display_name": MODEL_INFO[name]["display_name"],
                "family": MODEL_INFO[name]["family"],
                "supervised": MODEL_INFO[name]["supervised"],
                "note": MODEL_INFO[name].get("note"),
                **metrics_by_model[name],
            }
            for name in bundle_models
        },
    }
    metrics_path = models_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(f"Benchmark metrics saved to: {metrics_path}")

    return bundle_path


if __name__ == "__main__":
    train_all_models()
