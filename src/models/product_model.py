"""Task 4 by Tresor Shingiro Nkurunziza.

This file trains the product model that predicts which product category a customer will buy.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

MODEL_PATH = config.MODELS / "product_model.joblib"
TARGET = "product_category"

# We drop the id columns and the raw date so the model cannot just memorise them.
DROP = ["transaction_id", "customer_id_legacy", "customer_key", "purchase_date", TARGET]

CATEGORICAL = ["primary_platform", "amount_band"]


def load_data():
    """Read the merged dataset and split it into the input columns and the target column."""
    df = pd.read_csv(config.MERGED_CSV, parse_dates=["purchase_date"])
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df[TARGET]
    return df, X, y


def build_preprocessor(X):
    """Scale the number columns and one hot encode the text columns."""
    cat = [c for c in CATEGORICAL if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ]), num, cat


def candidates(pre):
    """Return the two models we want to compare."""
    return {
        "RandomForest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=2,
                class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1)),
        ]),
        "LogisticRegression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=config.RANDOM_STATE)),
        ]),
    }


def evaluate(model, X_te, y_te, classes):
    """Work out the accuracy, the F1 scores and the loss of a trained model on the test set."""
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "f1_weighted": f1_score(y_te, pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_te, proba, labels=list(classes)),
        "y_pred": pred,
    }


def train():
    """Train both models, compare them, print the scores and save the better one."""
    df, X, y = load_data()

    print("=" * 70)
    print("PRODUCT RECOMMENDATION MODEL")
    print("=" * 70)
    print(f"  rows      : {len(df)}")
    print(f"  features  : {X.shape[1]}")
    print(f"  target    : {TARGET} ({y.nunique()} classes)")
    print(f"  baseline  : {y.value_counts(normalize=True).max():.3f} "
          f"(always predict '{y.value_counts().idxmax()}')")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)
    print(f"  split     : {len(X_tr)} train / {len(X_te)} test (stratified)")

    pre, num, cat = build_preprocessor(X)
    print(f"  numeric   : {len(num)}   categorical (one hot): {cat}")

    classes = sorted(y.unique())
    results = {}

    for name, model in candidates(pre).items():
        model.fit(X_tr, y_tr)
        m = evaluate(model, X_te, y_te, classes)
        cv = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")
        m["cv_f1_mean"], m["cv_f1_std"] = float(cv.mean()), float(cv.std())
        results[name] = (model, m)

        print(f"\n  --- {name} ---")
        print(f"    Accuracy      : {m['accuracy']:.3f}")
        print(f"    F1 (weighted) : {m['f1_weighted']:.3f}")
        print(f"    F1 (macro)    : {m['f1_macro']:.3f}")
        print(f"    Log loss      : {m['log_loss']:.3f}")
        print(f"    5-fold CV F1  : {m['cv_f1_mean']:.3f} (+/- {m['cv_f1_std']:.3f})")

    best_name = max(results, key=lambda k: results[k][1]["f1_weighted"])
    best_model, best_metrics = results[best_name]

    print(f"\n  >>> selected: {best_name} because it has the highest weighted F1 on the test set")
    print(f"\n  Per-class report for {best_name}:")
    print(classification_report(y_te, best_metrics["y_pred"], zero_division=0, digits=3))

    cm = confusion_matrix(y_te, best_metrics["y_pred"], labels=classes)
    print("  Confusion matrix (rows = true, cols = predicted):")
    print("    " + "  ".join(f"{c[:11]:>11}" for c in classes))
    for row_label, row in zip(classes, cm):
        print(f"    {row_label[:11]:>11} " + "  ".join(f"{v:>11d}" for v in row))

    if best_name == "RandomForest":
        rf = best_model.named_steps["clf"]
        names = best_model.named_steps["pre"].get_feature_names_out()
        top = pd.Series(rf.feature_importances_, index=names).nlargest(12)
        print("\n  Top 12 features by importance:")
        for f, v in top.items():
            print(f"    {v:.4f}  {f}")

    joblib.dump({"model": best_model, "name": best_name, "classes": classes,
                 "feature_cols": list(X.columns)}, MODEL_PATH)
    print(f"\n  saved -> {MODEL_PATH}")

    clean = {k: v for k, v in best_metrics.items() if k != "y_pred"}
    return best_model, {"selected": best_name, **clean}


def load():
    if not MODEL_PATH.exists():
        raise SystemExit("Product model not trained yet. Run: python -m src.train_all")
    return joblib.load(MODEL_PATH)


def recommend_for_customer(customer_key: str):
    """Predict the next product category for one customer using their most recent purchase row."""
    bundle = load()
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df = pd.read_csv(config.MERGED_CSV, parse_dates=["purchase_date"])
    rows = df[df["customer_key"] == customer_key]
    if rows.empty:
        raise ValueError(f"Unknown customer: {customer_key}")

    latest = rows.sort_values("purchase_date").iloc[[-1]]
    X = latest[feature_cols]

    proba = model.predict_proba(X)[0]
    classes = model.named_steps["clf"].classes_
    order = np.argsort(proba)[::-1]
    return classes[order[0]], float(proba[order[0]]), dict(zip(classes, proba.astype(float)))


if __name__ == "__main__":
    train()
