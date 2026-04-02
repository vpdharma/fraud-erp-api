import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.services.features import (
    MODEL_FEATURE_COLUMNS,
    add_engineered_features,
    save_featured_dataframe,
    split_features_and_target,
)
from app.services.preprocess import (
    load_and_preprocess_csv,
    save_processed_dataframe,
)


def train_baseline_model() -> Path:
    """
    Train a beginner-friendly baseline fraud detection model.

    Steps:
    1. Load raw CSV
    2. Clean it
    3. Create fraud-related features
    4. Split into training and testing data
    5. Train Random Forest
    6. Evaluate the model
    7. Save the trained model as a .pkl file
    """
    project_root = PROJECT_ROOT
    raw_csv_path = project_root / "data" / "raw" / "orders.csv"

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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predicted_labels = model.predict(X_test)
    predicted_probabilities = model.predict_proba(X_test)[:, 1]

    print("Baseline model training completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Cleaned data saved to: {cleaned_output_path}")
    print(f"Feature-engineered data saved to: {featured_output_path}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predicted_labels))
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_labels, digits=4))

    try:
        roc_auc = roc_auc_score(y_test, predicted_probabilities)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except ValueError:
        print("ROC-AUC could not be calculated for this run.")

    model_artifact = {
        "model": model,
        "feature_columns": MODEL_FEATURE_COLUMNS,
    }

    model_output_path = project_root / "models" / "fraud_model.pkl"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_output_path)

    print(f"\nModel saved to: {model_output_path}")
    return model_output_path


if __name__ == "__main__":
    train_baseline_model()
