"""Task 4 by Tresor Shingiro Nkurunziza.

This file trains the face model and predicts which team member a face belongs to.
"""
import cv2
import joblib
import numpy as np

from src import config
from src.image_pipeline import extract_features
from src.models import biometric

MODEL_PATH = config.MODELS / "face_model.joblib"


def train():
    return biometric.train(config.IMAGE_FEATURES_CSV, "face_model", "facial recognition")


def load():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Face model not trained yet. Run: python -m src.train_all")
    return joblib.load(MODEL_PATH)


def predict_image(image_path):
    """Read a face image and return the best member, how sure the model is and all the scores."""
    bundle = load()
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    feats = extract_features(img)
    x = np.array([[feats[c] for c in feature_cols]], dtype=np.float64)

    proba = model.predict_proba(x)[0]
    classes = model.named_steps["clf"].classes_
    idx = int(np.argmax(proba))
    return classes[idx], float(proba[idx]), dict(zip(classes, proba.astype(float)))


if __name__ == "__main__":
    train()
