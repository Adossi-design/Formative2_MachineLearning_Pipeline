"""Exploratory data analysis by Thierry Alain Tresor Ibyishaka.

This file explores the merged dataset and saves the summary statistics and the four figures.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")
PALETTE = "deep"


def summarise(df: pd.DataFrame) -> None:
    """Print the column types and the summary statistics for the numbers and the categories."""
    print("=" * 70)
    print("VARIABLE TYPES")
    print("=" * 70)
    types = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(),
            "n_missing": df.isna().sum(),
            "role": [
                "target" if c == "product_category"
                else "identifier" if c in ("transaction_id", "customer_key", "customer_id_legacy")
                else "categorical" if df[c].dtype == "object"
                else "numeric"
                for c in df.columns
            ],
        }
    )
    print(types.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (numeric)")
    print("=" * 70)
    num = df.select_dtypes("number").drop(columns=["transaction_id", "customer_id_legacy"], errors="ignore")
    print(num.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())

    print("\n" + "=" * 70)
    print("SUMMARY (categorical)")
    print("=" * 70)
    for c in df.select_dtypes("object").columns:
        if df[c].nunique() <= 12:
            print(f"\n{c}:")
            print(df[c].value_counts().to_string())


def plot_distributions(df: pd.DataFrame) -> None:
    """First figure that shows how the target and the main number columns are spread out."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Distributions: target and primary numeric features", fontsize=14, weight="bold")

    order = df["product_category"].value_counts().index
    sns.countplot(data=df, y="product_category", order=order, ax=axes[0, 0], palette=PALETTE, hue="product_category", legend=False)
    axes[0, 0].set_title("Target: product_category (roughly balanced, 25 to 35 per class)")
    axes[0, 0].set_xlabel("transactions")

    sns.histplot(df["purchase_amount"], bins=20, kde=True, ax=axes[0, 1], color="#4C72B0")
    axes[0, 1].set_title("purchase_amount is spread wide from 50 to 500")
    axes[0, 1].set_xlabel("purchase amount")

    sns.histplot(df["engagement_mean"], bins=20, kde=True, ax=axes[1, 0], color="#55A868")
    axes[1, 0].set_title("engagement_mean per customer from social")
    axes[1, 0].set_xlabel("mean engagement score")

    sns.histplot(df["purchase_interest_mean"], bins=20, kde=True, ax=axes[1, 1], color="#C44E52")
    axes[1, 1].set_title("purchase_interest_mean per customer from social")
    axes[1, 1].set_xlabel("mean purchase interest")

    plt.tight_layout()
    out = config.FIGURES / "eda_01_distributions.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def plot_outliers(df: pd.DataFrame) -> None:
    """Second figure that looks for outliers inside each product category."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Outlier inspection", fontsize=14, weight="bold")

    sns.boxplot(data=df, x="product_category", y="purchase_amount", ax=axes[0], palette=PALETTE, hue="product_category", legend=False)
    axes[0].set_title("purchase_amount by category")
    axes[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=df, x="product_category", y="customer_rating", ax=axes[1], palette=PALETTE, hue="product_category", legend=False)
    axes[1].set_title("customer_rating by category")
    axes[1].tick_params(axis="x", rotation=30)

    sns.boxplot(data=df, y="amount_zscore", ax=axes[2], color="#8172B2")
    axes[2].set_title("amount_zscore, purchase against the customer own average")
    axes[2].axhline(0, ls="--", c="grey", lw=1)

    plt.tight_layout()
    out = config.FIGURES / "eda_02_outliers.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def plot_correlations(df: pd.DataFrame) -> None:
    """Third figure that shows how strongly the number columns move together."""
    cols = [
        "purchase_amount", "customer_rating", "engagement_mean", "purchase_interest_mean",
        "sentiment_mean", "n_platforms", "customer_txn_count", "customer_avg_amount",
        "customer_total_spend", "amount_vs_customer_avg", "amount_zscore",
        "engagement_x_interest", "spend_per_engagement", "rating_vs_sentiment_gap",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    plt.title("Correlation matrix of the source and engineered features", fontsize=13, weight="bold")
    plt.tight_layout()
    out = config.FIGURES / "eda_03_correlations.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def plot_cross_source(df: pd.DataFrame) -> None:
    """Fourth figure that asks if the social columns can actually separate the target."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cross source signal: do social features separate product_category?",
                 fontsize=14, weight="bold")

    sns.violinplot(data=df, x="product_category", y="purchase_interest_mean",
                   ax=axes[0], palette=PALETTE, hue="product_category", legend=False)
    axes[0].set_title("purchase_interest by category")
    axes[0].tick_params(axis="x", rotation=30)

    sns.scatterplot(data=df, x="engagement_mean", y="purchase_amount",
                    hue="product_category", ax=axes[1], palette=PALETTE, s=45)
    axes[1].set_title("engagement against spend, coloured by category")
    axes[1].legend(fontsize=7, title=None)

    ct = pd.crosstab(df["primary_platform"], df["product_category"], normalize="index")
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=axes[2], cbar=False, annot_kws={"size": 8})
    axes[2].set_title("category mix by primary platform")

    plt.tight_layout()
    out = config.FIGURES / "eda_04_cross_source.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def run():
    df = pd.read_csv(config.MERGED_CSV)
    summarise(df)
    print("\n" + "=" * 70)
    print("FIGURES")
    print("=" * 70)
    plot_distributions(df)
    plot_outliers(df)
    plot_correlations(df)
    plot_cross_source(df)
    return df


if __name__ == "__main__":
    run()
