from pathlib import Path

import pandas as pd


REQUIRED_RAW_COLUMNS = [
    "order_id",
    "customer_id",
    "product_id",
    "order_amount",
    "quantity",
    "discount_percent",
    "billing_state",
    "shipping_state",
    "account_age_days",
    "customer_order_count",
    "customer_return_count",
    "refund_amount",
    "return_flag",
    "urgent_shipping_flag",
    "manual_override_flag",
    "order_hour",
    "fraud_label",
]

NUMERIC_COLUMNS = [
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
    "fraud_label",
]

TEXT_COLUMNS = [
    "order_id",
    "customer_id",
    "product_id",
    "billing_state",
    "shipping_state",
]


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names easier to work with by:
    - converting to lowercase
    - trimming extra spaces
    - replacing spaces with underscores
    """
    cleaned_dataframe = dataframe.copy()
    cleaned_dataframe.columns = [
        column.strip().lower().replace(" ", "_")
        for column in cleaned_dataframe.columns
    ]
    return cleaned_dataframe


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Stop early with a clear message if important columns are missing.
    """
    missing_columns = [
        column for column in REQUIRED_RAW_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")


def remove_duplicate_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows so the same order is not counted twice.
    """
    return dataframe.drop_duplicates().reset_index(drop=True)


def convert_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric-looking columns into real numeric columns.
    Invalid values become missing values first and will be filled later.
    """
    cleaned_dataframe = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        cleaned_dataframe[column] = pd.to_numeric(
            cleaned_dataframe[column], errors="coerce"
        )

    return cleaned_dataframe


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with simple beginner-friendly rules:
    - numeric columns -> median
    - text columns -> 'unknown'
    """
    cleaned_dataframe = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        median_value = cleaned_dataframe[column].median()
        if pd.isna(median_value):
            median_value = 0
        cleaned_dataframe[column] = cleaned_dataframe[column].fillna(median_value)

    for column in TEXT_COLUMNS:
        cleaned_dataframe[column] = cleaned_dataframe[column].fillna("unknown")

    return cleaned_dataframe


def preprocess_orders_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing function used by later scripts and APIs.
    """
    cleaned_dataframe = standardize_column_names(dataframe)
    validate_required_columns(cleaned_dataframe)
    cleaned_dataframe = remove_duplicate_rows(cleaned_dataframe)
    cleaned_dataframe = convert_numeric_columns(cleaned_dataframe)
    cleaned_dataframe = fill_missing_values(cleaned_dataframe)
    return cleaned_dataframe


def load_and_preprocess_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file and apply preprocessing in one step.
    """
    dataframe = pd.read_csv(csv_path)
    return preprocess_orders_dataframe(dataframe)


def save_processed_dataframe(
    dataframe: pd.DataFrame,
    output_filename: str = "orders_cleaned.csv",
) -> Path:
    """
    Save cleaned data into the data/processed folder.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "processed" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path
