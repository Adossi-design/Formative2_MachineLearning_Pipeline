"""Task 4 by Tresor Shingiro Nkurunziza.

This file trains the voice model and predicts which team member a voice belongs to.
"""
import warnings

import joblib
import librosa
import numpy as np

from src import config
from src.audio_pipeline import extract_features
from src.models import biometric

MODEL_PATH = config.MODELS / "voice_model.joblib"


def train():
    return biometric.train(config.AUDIO_FEATURES_CSV, "voice_model", "voiceprint verification")


def load():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Voice model not trained yet. Run: python -m src.train_all")
    return joblib.load(MODEL_PATH)


def predict_audio(audio_path):
    """Read a voice clip and return the best member, how sure the model is and all the scores."""
    bundle = load()
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
        if len(y) < sr * 0.3:
            raise ValueError(f"Clip too short to verify: {audio_path}")

        feats = extract_features(y, sr)
    x = np.array([[feats[c] for c in feature_cols]], dtype=np.float64)

    proba = model.predict_proba(x)[0]
    classes = model.named_steps["clf"].classes_
    idx = int(np.argmax(proba))
    return classes[idx], float(proba[idx]), dict(zip(classes, proba.astype(float)))


if __name__ == "__main__":
    train()
