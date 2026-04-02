from pathlib import Path

import numpy as np
import pandas as pd


def build_sample_dataset(row_count: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic ERP-like order data for a fraud detection project.

    We use simple business-style rules so the dataset is easier to explain in a viva:
    - very high order value can be suspicious
    - billing and shipping mismatch can be suspicious
    - high discount can be suspicious
    - urgent shipping + manual override can be suspicious
    - very new customer with high value order can be suspicious
    """
    rng = np.random.default_rng(random_seed)

    states = [
        "California",
        "Texas",
        "New York",
        "Florida",
        "Illinois",
        "Washington",
        "Arizona",
        "Georgia",
    ]

    customer_ids = [f"CUST{index:04d}" for index in range(1, 301)]
    product_ids = [f"PROD{index:04d}" for index in range(1, 81)]

    customer_id = rng.choice(customer_ids, size=row_count)
    product_id = rng.choice(product_ids, size=row_count)

    quantity = rng.integers(1, 25, size=row_count)
    base_unit_price = rng.uniform(20, 600, size=row_count)
    discount_percent = rng.choice(
        [0, 5, 10, 15, 20, 25, 30, 40],
        size=row_count,
        p=[0.15, 0.18, 0.20, 0.15, 0.12, 0.08, 0.07, 0.05],
    )

    gross_amount = quantity * base_unit_price
    order_amount = gross_amount * (1 - (discount_percent / 100))
    order_amount = np.round(order_amount, 2)

    billing_state = rng.choice(states, size=row_count)
    shipping_state = billing_state.copy()

    mismatch_mask = rng.random(row_count) < 0.18
    replacement_states = rng.choice(states, size=mismatch_mask.sum())
    shipping_state[mismatch_mask] = replacement_states

    # Make sure mismatch rows really have different states.
    same_state_mask = shipping_state == billing_state
    forced_mismatch_mask = mismatch_mask & same_state_mask
    if forced_mismatch_mask.any():
        billing_forced = billing_state[forced_mismatch_mask]
        new_states = []
        for state in billing_forced:
            other_states = [item for item in states if item != state]
            new_states.append(rng.choice(other_states))
        shipping_state[forced_mismatch_mask] = new_states

    account_age_days = rng.integers(1, 2000, size=row_count)
    customer_order_count = rng.integers(1, 120, size=row_count)
    customer_return_count = np.minimum(
        rng.integers(0, 30, size=row_count),
        customer_order_count,
    )

    return_flag = (rng.random(row_count) < 0.22).astype(int)
    refund_amount = np.where(
        return_flag == 1,
        order_amount * rng.uniform(0.3, 1.1, size=row_count),
        0,
    )
    refund_amount = np.round(refund_amount, 2)

    urgent_shipping_flag = (rng.random(row_count) < 0.25).astype(int)
    manual_override_flag = (rng.random(row_count) < 0.10).astype(int)
    order_hour = rng.integers(0, 24, size=row_count)

    shipping_mismatch_flag = (billing_state != shipping_state).astype(int)
    high_discount_flag = (discount_percent >= 25).astype(int)
    new_customer_high_value_flag = (
        (account_age_days < 60) & (order_amount > 5000)
    ).astype(int)
    high_refund_flag = (refund_amount > order_amount * 0.8).astype(int)
    return_ratio = customer_return_count / np.maximum(customer_order_count, 1)
    high_return_ratio_flag = (return_ratio > 0.35).astype(int)
    odd_hour_flag = ((order_hour <= 5) | (order_hour >= 23)).astype(int)

    # Build a simple fraud score from domain-style rules plus a little randomness.
    fraud_score = (
        shipping_mismatch_flag * 2.0
        + high_discount_flag * 1.5
        + new_customer_high_value_flag * 2.5
        + high_refund_flag * 1.7
        + high_return_ratio_flag * 1.6
        + (urgent_shipping_flag & manual_override_flag) * 2.2
        + odd_hour_flag * 0.6
        + (order_amount > 7000).astype(int) * 1.2
        + rng.normal(0, 0.7, size=row_count)
    )

    fraud_label = (fraud_score >= 4.0).astype(int)

    dataframe = pd.DataFrame(
        {
            "order_id": [f"ORD{index:06d}" for index in range(1, row_count + 1)],
            "customer_id": customer_id,
            "product_id": product_id,
            "order_amount": order_amount,
            "quantity": quantity,
            "discount_percent": discount_percent,
            "billing_state": billing_state,
            "shipping_state": shipping_state,
            "account_age_days": account_age_days,
            "customer_order_count": customer_order_count,
            "customer_return_count": customer_return_count,
            "refund_amount": refund_amount,
            "return_flag": return_flag,
            "urgent_shipping_flag": urgent_shipping_flag,
            "manual_override_flag": manual_override_flag,
            "order_hour": order_hour,
            "fraud_label": fraud_label,
        }
    )

    return dataframe


def save_dataset(dataframe: pd.DataFrame) -> Path:
    """
    Save the generated dataset into the project's raw data folder.
    """
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "raw" / "orders.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    dataframe = build_sample_dataset(row_count=1000, random_seed=42)
    output_path = save_dataset(dataframe)

    suspicious_count = int(dataframe["fraud_label"].sum())
    total_count = len(dataframe)

    print("Sample ERP order dataset created successfully.")
    print(f"Saved file: {output_path}")
    print(f"Total rows: {total_count}")
    print(f"Suspicious rows (fraud_label = 1): {suspicious_count}")
    print("\nPreview:")
    print(dataframe.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
