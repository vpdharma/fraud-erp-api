from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FEATURE_COLUMNS = [
    "order_amount",
    "quantity",
    "discount_percent",
    "account_age_days",
    "customer_order_count",
    "customer_return_count",
    "refund_amount",
    "return_flag",
    "urgent_shipping_flag",
    "manual_override_flag",
    "order_hour",
    "shipping_mismatch_flag",
    "high_discount_flag",
    "new_customer_high_value_flag",
    "refund_to_order_ratio",
    "customer_return_ratio",
    "urgent_manual_override_flag",
    "odd_order_hour_flag",
    "high_order_amount_flag",
]


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create fraud-related features from ERP transaction fields.

    These are business-style features that are easy to explain:
    - shipping mismatch
    - high discount
    - new customer with high order value
    - refund compared to order value
    - customer return ratio
    - urgent shipping with manual override
    """
    featured_dataframe = dataframe.copy()

    featured_dataframe["shipping_mismatch_flag"] = (
        featured_dataframe["billing_state"].astype(str).str.strip().str.lower()
        != featured_dataframe["shipping_state"].astype(str).str.strip().str.lower()
    ).astype(int)

    featured_dataframe["high_discount_flag"] = (
        featured_dataframe["discount_percent"] >= 25
    ).astype(int)

    featured_dataframe["new_customer_high_value_flag"] = (
        (featured_dataframe["account_age_days"] < 60)
        & (featured_dataframe["order_amount"] > 5000)
    ).astype(int)

    featured_dataframe["refund_to_order_ratio"] = np.where(
        featured_dataframe["order_amount"] > 0,
        featured_dataframe["refund_amount"] / featured_dataframe["order_amount"],
        0,
    )

    featured_dataframe["customer_return_ratio"] = np.where(
        featured_dataframe["customer_order_count"] > 0,
        featured_dataframe["customer_return_count"]
        / featured_dataframe["customer_order_count"],
        0,
    )

    featured_dataframe["urgent_manual_override_flag"] = (
        (featured_dataframe["urgent_shipping_flag"] == 1)
        & (featured_dataframe["manual_override_flag"] == 1)
    ).astype(int)

    featured_dataframe["odd_order_hour_flag"] = (
        (featured_dataframe["order_hour"] <= 5)
        | (featured_dataframe["order_hour"] >= 23)
    ).astype(int)

    featured_dataframe["high_order_amount_flag"] = (
        featured_dataframe["order_amount"] > 7000
    ).astype(int)

    featured_dataframe["refund_to_order_ratio"] = featured_dataframe[
        "refund_to_order_ratio"
    ].round(4)
    featured_dataframe["customer_return_ratio"] = featured_dataframe[
        "customer_return_ratio"
    ].round(4)

    return featured_dataframe


def get_model_input_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns we want to send into the machine learning model.
    """
    missing_columns = [
        column for column in MODEL_FEATURE_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing engineered feature columns: {missing_text}")

    return dataframe[MODEL_FEATURE_COLUMNS].copy()


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = "fraud_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model input columns (X) and target labels (y).

    X = the columns used by the model to learn patterns
    y = the correct answer column we want to predict
    """
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    feature_dataframe = get_model_input_dataframe(dataframe)
    target_series = dataframe[target_column].astype(int)
    return feature_dataframe, target_series


def save_featured_dataframe(
    dataframe: pd.DataFrame,
    output_filename: str = "orders_featured.csv",
) -> Path:
    """
    Save feature-engineered data into the data/processed folder.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "processed" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path
