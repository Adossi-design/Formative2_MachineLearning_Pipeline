"""Task 1 by Thierry Alain Tresor Ibyishaka.

This file cleans the two datasets, joins them on a shared id, and builds the engineered columns.
"""
import numpy as np
import pandas as pd

from src import config

SENTIMENT_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def load_raw():
    social = pd.read_csv(config.SOCIAL_CSV)
    tx = pd.read_csv(config.TRANSACTIONS_CSV)
    return social, tx


def audit(df: pd.DataFrame, name: str) -> dict:
    """Look at a table before we clean it and count the problems we can see."""
    return {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "null_cells": int(df.isna().sum().sum()),
        "nulls_by_column": df.isna().sum()[lambda s: s > 0].to_dict(),
    }


def clean_social(social: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated rows, fix the data types and keep the scores inside their real range."""
    social = social.drop_duplicates().copy()

    social["engagement_score"] = pd.to_numeric(social["engagement_score"], errors="coerce")
    social["purchase_interest_score"] = pd.to_numeric(
        social["purchase_interest_score"], errors="coerce"
    )
    # Engagement should be between 0 and 100 and interest should be between 1 and 5.
    social["engagement_score"] = social["engagement_score"].clip(0, 100)
    social["purchase_interest_score"] = social["purchase_interest_score"].clip(1, 5)

    social["review_sentiment"] = social["review_sentiment"].str.strip().str.title()
    social["sentiment_value"] = social["review_sentiment"].map(SENTIMENT_MAP)

    social = social.dropna(subset=["customer_id_new", "engagement_score"])
    return social


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Fix the data types, read the dates and fill in the ratings that are missing."""
    tx = tx.drop_duplicates().copy()

    tx["purchase_amount"] = pd.to_numeric(tx["purchase_amount"], errors="coerce")
    tx["customer_rating"] = pd.to_numeric(tx["customer_rating"], errors="coerce")
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"], errors="coerce")
    tx["product_category"] = tx["product_category"].str.strip().str.title()

    # About ten ratings are missing, and because ratings are not the same across the
    # different product categories we fill each missing value with the middle rating
    # of its own category, which is a better guess than one single number for all rows.
    tx["rating_was_imputed"] = tx["customer_rating"].isna().astype(int)
    tx["customer_rating"] = tx.groupby("product_category")["customer_rating"].transform(
        lambda s: s.fillna(s.median())
    )
    tx["customer_rating"] = tx["customer_rating"].fillna(tx["customer_rating"].median())

    tx = tx.dropna(subset=["purchase_amount", "purchase_date", "product_category"])
    return tx


def build_join_key(social: pd.DataFrame, tx: pd.DataFrame):
    """Make one shared id column so the two tables can be joined together.

    The transactions file writes the customer id as a plain number like 100, and the
    social file writes the same customer as A100, so the two files cannot join the way
    they are, and to fix this we make a new column called customer_key that looks like
    A100 in both tables so that the join can find the matching customers.
    """
    tx = tx.copy()
    social = social.copy()
    tx["customer_key"] = "A" + tx["customer_id_legacy"].astype(int).astype(str)
    social["customer_key"] = social["customer_id_new"].astype(str).str.strip().str.upper()
    return social, tx


def aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """Turn the many social rows of one customer into a single row for that customer.

    One customer can appear many times in the social file because they appear once for
    every platform they use, and the transactions file has one row for every purchase, so
    if the two files were joined directly every purchase would be copied many times and the
    result would be incorrect, and to prevent this we first reduce the social data to one
    row per customer so that the join becomes a safe many to one join.
    """
    grouped = social.groupby("customer_key")

    agg = grouped.agg(
        engagement_mean=("engagement_score", "mean"),
        engagement_max=("engagement_score", "max"),
        purchase_interest_mean=("purchase_interest_score", "mean"),
        purchase_interest_max=("purchase_interest_score", "max"),
        sentiment_mean=("sentiment_value", "mean"),
        n_platforms=("social_media_platform", "nunique"),
        n_social_records=("social_media_platform", "size"),
    )

    # The main platform is the one where the customer has the highest engagement.
    primary = (
        social.sort_values("engagement_score", ascending=False)
        .groupby("customer_key")["social_media_platform"]
        .first()
        .rename("primary_platform")
    )

    # For each customer we work out the share of positive, neutral and negative reviews.
    sent = (
        social.groupby(["customer_key", "review_sentiment"])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda d: d.div(d.sum(axis=1), axis=0))
        .add_prefix("sentiment_share_")
    )

    return agg.join(primary).join(sent).reset_index()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build new columns that mix the shopping behaviour together with the social behaviour."""
    df = df.sort_values(["customer_key", "purchase_date"]).copy()

    # Columns that come from the purchase date.
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

    # Columns that describe the whole shopping history of each customer.
    g = df.groupby("customer_key")["purchase_amount"]
    df["customer_txn_count"] = g.transform("size")
    df["customer_avg_amount"] = g.transform("mean")
    df["customer_total_spend"] = g.transform("sum")
    df["customer_amount_std"] = g.transform("std").fillna(0)

    # These two columns show whether a purchase is large or ordinary for that same customer.
    df["amount_vs_customer_avg"] = df["purchase_amount"] / df["customer_avg_amount"]
    df["amount_zscore"] = np.where(
        df["customer_amount_std"] > 0,
        (df["purchase_amount"] - df["customer_avg_amount"]) / df["customer_amount_std"],
        0.0,
    )

    # These columns combine the social data and the shopping data, which is the main purpose of the merge.
    df["engagement_x_interest"] = df["engagement_mean"] * df["purchase_interest_mean"]
    df["interest_per_platform"] = df["purchase_interest_mean"] / df["n_platforms"]
    # This shows the gap between how they rate a purchase and how they review it in public.
    df["rating_vs_sentiment_gap"] = df["customer_rating"] - (df["sentiment_mean"] + 2) * 1.25
    df["spend_per_engagement"] = df["purchase_amount"] / df["engagement_mean"].replace(0, np.nan)
    df["high_intent"] = (df["purchase_interest_mean"] >= 3.5).astype(int)

    df["amount_band"] = pd.cut(
        df["purchase_amount"],
        bins=[0, 150, 300, 400, np.inf],
        labels=["low", "medium", "high", "premium"],
    ).astype(str)

    return df


def validate_merge(tx: pd.DataFrame, merged: pd.DataFrame, social_agg: pd.DataFrame) -> dict:
    """Check that the join went well and did not lose rows or copy them by mistake."""
    matched = merged["engagement_mean"].notna().sum()
    checks = {
        "transactions_in": len(tx),
        "rows_out": len(merged),
        "row_count_preserved": len(tx) == len(merged),
        "no_fanout_duplicate_txn_ids": merged["transaction_id"].is_unique,
        "customers_in_transactions": tx["customer_key"].nunique(),
        "customers_in_social": social_agg["customer_key"].nunique(),
        "transactions_matched_to_social": int(matched),
        "match_rate": round(matched / len(merged), 4),
        "unmatched_customers": sorted(
            set(merged.loc[merged["engagement_mean"].isna(), "customer_key"])
        ),
        "remaining_nulls": int(merged.isna().sum().sum()),
    }
    return checks


def build(verbose: bool = True):
    social_raw, tx_raw = load_raw()

    audits = [audit(social_raw, "customer_social_profiles"), audit(tx_raw, "customer_transactions")]

    social = clean_social(social_raw)
    tx = clean_transactions(tx_raw)
    social, tx = build_join_key(social, tx)
    social_agg = aggregate_social(social)

    # We use a left join so that every purchase is kept even when the customer has no
    # social profile, because an inner join would throw away real purchases and change
    # how many rows each product category has.
    merged = tx.merge(social_agg, on="customer_key", how="left", validate="many_to_one")

    checks = validate_merge(tx, merged, social_agg)

    # For customers with no social data we fill the numbers with the middle value and add
    # a flag column, so the model can treat having no social profile as its own signal.
    merged["has_social_profile"] = merged["engagement_mean"].notna().astype(int)
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())
    merged["primary_platform"] = merged["primary_platform"].fillna("None")

    merged = engineer_features(merged)
    merged.to_csv(config.MERGED_CSV, index=False)

    if verbose:
        print("=" * 70)
        print("RAW DATA AUDIT")
        print("=" * 70)
        for a in audits:
            print(f"\n{a['dataset']}: {a['rows']} rows x {a['columns']} cols")
            print(f"  exact duplicate rows : {a['exact_duplicate_rows']}")
            print(f"  null cells           : {a['null_cells']} {a['nulls_by_column']}")

        print("\n" + "=" * 70)
        print("MERGE VALIDATION")
        print("=" * 70)
        for k, v in checks.items():
            print(f"  {k:<32}: {v}")

        print("\n" + "=" * 70)
        print(f"MERGED DATASET -> {config.MERGED_CSV}")
        print("=" * 70)
        print(f"  shape   : {merged.shape}")
        print(f"  target  : product_category")
        print(f"  classes :\n{merged['product_category'].value_counts().to_string()}")

    return merged, checks, audits


if __name__ == "__main__":
    build()
