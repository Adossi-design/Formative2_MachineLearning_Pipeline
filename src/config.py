"""Combined settings shared by the whole project.

This file keeps all the paths, the team names and the constants in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
IMAGES = DATA / "images"
AUDIO = DATA / "audio"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

for _d in (RAW, PROCESSED, IMAGES, AUDIO, MODELS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SOCIAL_CSV = RAW / "customer_social_profiles.csv"
TRANSACTIONS_CSV = RAW / "customer_transactions.csv"
MERGED_CSV = PROCESSED / "merged_dataset.csv"
IMAGE_FEATURES_CSV = PROCESSED / "image_features.csv"
AUDIO_FEATURES_CSV = PROCESSED / "audio_features.csv"

# The key is the folder name on disk and also the label the models learn.
TEAM = {
    "adossi_fred_william": "Adossi Fred William",
    "homere_singizwa": "Homere Singizwa",
    "thierry_alain_tresor_ibyishaka": "Thierry Alain Tresor Ibyishaka",
    "tresor_shingiro_nkurunziza": "Tresor Shingiro Nkurunziza",
}

# This label stands for any person who is not one of the team members.
IMPOSTOR = "unauthorized"

# The image formats that we allow a member to drop into their folder.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2", ".ppm")
# The audio formats we allow, but m4a and aac only work when ffmpeg is installed.
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif", ".au", ".m4a", ".aac", ".opus", ".wma")


def list_media(folder, exts):
    """Return every file in the folder whose ending matches one of the endings we accept."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


EXPRESSIONS = ["neutral", "smiling", "surprised"]
PHRASES = {
    "yes_approve": "Yes, approve",
    "confirm_transaction": "Confirm transaction",
}

FACE_SIZE = (128, 128)   # every image is made this size before we read features
SAMPLE_RATE = 22050      # every audio clip is changed to this sample rate
N_MFCC = 20

RANDOM_STATE = 42

# If the model is less sure than this value we treat it as not that person.
FACE_THRESHOLD = 0.60
VOICE_THRESHOLD = 0.60
