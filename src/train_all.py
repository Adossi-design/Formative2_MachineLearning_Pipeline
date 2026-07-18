"""Combined file that runs the whole project in one command.

This file standardises the media, merges the data, builds the features and trains the three models.
"""
import json
import warnings

warnings.filterwarnings("ignore")

from src import audio_pipeline, config, data_merge, image_pipeline, normalize_media
from src.models import face_model, product_model, voice_model


def main():
    print("\n" + "#" * 70)
    print("#  STEP 1 of 5, STANDARDISE THE MEDIA")
    print("#" * 70 + "\n")
    n = normalize_media.run()
    if n == 0:
        print("  every file is already a .jpeg image or a .wav clip")

    print("\n" + "#" * 70)
    print("#  STEP 2 of 5, MERGE TABULAR DATA")
    print("#" * 70 + "\n")
    data_merge.build()

    print("\n" + "#" * 70)
    print("#  STEP 3 of 5, IMAGE FEATURES")
    print("#" * 70 + "\n")
    image_pipeline.run()

    print("\n" + "#" * 70)
    print("#  STEP 4 of 5, AUDIO FEATURES")
    print("#" * 70 + "\n")
    audio_pipeline.run()

    print("\n" + "#" * 70)
    print("#  STEP 5 of 5, TRAIN THE THREE MODELS")
    print("#" * 70 + "\n")

    _, face_metrics = face_model.train()
    print()
    _, voice_metrics = voice_model.train()
    print()
    _, product_metrics = product_model.train()

    metrics = {
        "face_recognition": face_metrics,
        "voiceprint_verification": voice_metrics,
        "product_recommendation": product_metrics,
    }
    out = config.ROOT / "reports" / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))

    print("\n" + "#" * 70)
    print("#  SUMMARY")
    print("#" * 70)
    print(f"\n  {'model':<26}{'accuracy':>10}{'F1':>10}{'log loss':>11}")
    print("  " + "-" * 57)
    print(f"  {'Facial recognition':<26}{face_metrics['accuracy_mean']:>10.3f}"
          f"{face_metrics['f1_weighted_mean']:>10.3f}{face_metrics['log_loss_mean']:>11.3f}")
    print(f"  {'Voiceprint verification':<26}{voice_metrics['accuracy_mean']:>10.3f}"
          f"{voice_metrics['f1_weighted_mean']:>10.3f}{voice_metrics['log_loss_mean']:>11.3f}")
    print(f"  {'Product recommendation':<26}{product_metrics['accuracy']:>10.3f}"
          f"{product_metrics['f1_weighted']:>10.3f}{product_metrics['log_loss']:>11.3f}")
    print(f"\n  metrics -> {out}")
    print("\n  Next:  python -m app.cli_app --demo\n")


if __name__ == "__main__":
    main()
