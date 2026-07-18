"""Task 4 by Tresor Shingiro Nkurunziza.

This combined file is the shared trainer that both the face model and the voice model use.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

META_COLS = {
    "member", "member_name", "is_authorized", "expression", "phrase", "phrase_text",
    "augmentation", "source_file",
}


def load_features(csv_path):
    """Read a feature file and split it into the numbers, the labels and the file groups."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["member"].to_numpy()
    groups = df["source_file"].to_numpy()
    return df, X, y, feature_cols, groups


def build_model():
    """Build a pipeline that scales the numbers and then trains a random forest."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def cross_validate(X, y, groups, n_splits=3):
    """Test the model with grouped folds and give back the average scores."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    classes = np.unique(y)

    accs, f1s, losses = [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in cv.split(X, y, groups):
        model = build_model()
        model.fit(X[train_idx], y[train_idx])

        pred = model.predict(X[test_idx])
        proba = model.predict_proba(X[test_idx])

        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="weighted", zero_division=0))
        # We pass the class list so the loss is correct even if a fold misses a class.
        losses.append(log_loss(y[test_idx], proba, labels=list(model.named_steps["clf"].classes_)))

        all_true.extend(y[test_idx])
        all_pred.extend(pred)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_weighted_mean": float(np.mean(f1s)),
        "f1_weighted_std": float(np.std(f1s)),
        "log_loss_mean": float(np.mean(losses)),
        "log_loss_std": float(np.std(losses)),
        "n_folds": n_splits,
        "y_true": all_true,
        "y_pred": all_pred,
        "classes": list(classes),
    }


def train(csv_path, model_name: str, label: str, n_splits=3):
    """Train one biometric model, print its scores and save it to disk."""
    df, X, y, feature_cols, groups = load_features(csv_path)

    print("=" * 70)
    print(f"{label.upper()} MODEL, Random Forest")
    print("=" * 70)
    print(f"  rows            : {len(df)}  ({df['source_file'].nunique()} source files x "
          f"{len(df) // max(df['source_file'].nunique(), 1)} variants)")
    print(f"  features        : {len(feature_cols)}")
    print(f"  classes         : {list(np.unique(y))}")

    cv = cross_validate(X, y, groups, n_splits)

    print(f"\n  Grouped {cv['n_folds']}-fold CV with no augmentation leakage:")
    print(f"    Accuracy      : {cv['accuracy_mean']:.3f}  (+/- {cv['accuracy_std']:.3f})")
    print(f"    F1 (weighted) : {cv['f1_weighted_mean']:.3f}  (+/- {cv['f1_weighted_std']:.3f})")
    print(f"    Log loss      : {cv['log_loss_mean']:.3f}  (+/- {cv['log_loss_std']:.3f})")

    print(f"\n  Per-class report pooled across folds:")
    print(classification_report(cv["y_true"], cv["y_pred"], zero_division=0, digits=3))

    cm = confusion_matrix(cv["y_true"], cv["y_pred"], labels=cv["classes"])
    print("  Confusion matrix (rows = true, cols = predicted):")
    print("    " + "  ".join(f"{c[:9]:>9}" for c in cv["classes"]))
    for row_label, row in zip(cv["classes"], cm):
        print(f"    {row_label[:9]:>9} " + "  ".join(f"{v:>9d}" for v in row))

    # We report the folds above, but the model we save is trained on all of the data.
    final = build_model()
    final.fit(X, y)

    path = config.MODELS / f"{model_name}.joblib"
    import joblib
    joblib.dump({"model": final, "feature_cols": feature_cols, "classes": list(final.named_steps["clf"].classes_)}, path)
    print(f"\n  saved -> {path}")

    metrics = {k: v for k, v in cv.items() if k not in ("y_true", "y_pred", "classes")}
    return final, metrics
