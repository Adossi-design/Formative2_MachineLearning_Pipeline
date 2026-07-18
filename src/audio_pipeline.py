"""Task 3 by Adossi Fred William.

This file loads the voice clips, augments them, and saves their features to audio_features.csv.
"""
import matplotlib

matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

RNG = np.random.default_rng(config.RANDOM_STATE)
SR = config.SAMPLE_RATE


# The functions below are the four ways we change a voice clip.
def aug_pitch_shift(y, sr, steps=2.0):
    """Move the pitch up by two notes so the voice still works when the speaker sounds excited."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)


def aug_time_stretch(y, sr, rate=1.15):
    """Make the clip play faster so the voice still works when the person speaks quickly."""
    return librosa.effects.time_stretch(y=y, rate=rate)


def aug_background_noise(y, sr, snr_db=15.0):
    """Add some soft noise so the voice still works inside a noisy room."""
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10)) if sig_power > 0 else 0
    return y + RNG.normal(0, np.sqrt(noise_power), len(y)).astype(np.float32)


def aug_gain(y, sr, gain=0.6):
    """Make the clip quieter so the voice still works when the person is far from the mic."""
    return y * gain


AUGMENTATIONS = {
    "original": lambda y, sr: y,
    "pitch_shift": aug_pitch_shift,
    "time_stretch": aug_time_stretch,
    "background_noise": aug_background_noise,
    "gain_down": aug_gain,
}


def extract_features(y: np.ndarray, sr: int) -> dict:
    """Read the MFCCs, the roll off, the energy and a few more sound descriptors from a clip."""
    feats = {}

    # MFCCs are the main voice fingerprint, and the mean and std make a fixed size list.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    for i in range(config.N_MFCC):
        feats[f"mfcc_{i:02d}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i:02d}_std"] = float(mfcc[i].std())

    # The delta MFCCs show how the sound changes over time and not only its average.
    d = librosa.feature.delta(mfcc)
    for i in range(config.N_MFCC):
        feats[f"mfcc_delta_{i:02d}_mean"] = float(d[i].mean())

    # Spectral roll off is the frequency under which most of the energy sits.
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    feats["rolloff_mean"] = float(roll.mean())
    feats["rolloff_std"] = float(roll.std())

    # The energy of the clip.
    rms = librosa.feature.rms(y=y)
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    feats["rms_max"] = float(rms.max())

    # A few more descriptors about the tone of the voice.
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats["centroid_mean"] = float(cent.mean())
    feats["centroid_std"] = float(cent.std())

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats["bandwidth_mean"] = float(bw.mean())

    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())

    flat = librosa.feature.spectral_flatness(y=y)
    feats["flatness_mean"] = float(flat.mean())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i:02d}"] = float(chroma[i].mean())

    # The base pitch of the voice, which is very different from one person to another.
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    feats["f0_mean"] = float(np.mean(f0)) if len(f0) else 0.0
    feats["f0_std"] = float(np.std(f0)) if len(f0) else 0.0

    feats["duration_sec"] = float(len(y) / sr)
    return feats


def plot_waveforms_and_spectrograms(clips: dict) -> None:
    """Draw the waveform and the mel spectrogram of one clip for each member."""
    members = list(clips)
    fig, axes = plt.subplots(len(members), 2, figsize=(13, 2.7 * len(members)))
    axes = np.atleast_2d(axes)
    fig.suptitle('Voice samples, waveform and mel spectrogram of "Yes, approve"',
                 fontsize=14, weight="bold")

    for r, slug in enumerate(members):
        y, sr = clips[slug]
        name = config.TEAM.get(slug, "Unauthorized / impostor")

        librosa.display.waveshow(y, sr=sr, ax=axes[r, 0], color="#4C72B0")
        axes[r, 0].set_title(f"{name} waveform", fontsize=10)
        axes[r, 0].set_ylabel("amplitude", fontsize=8)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[r, 1], cmap="magma")
        axes[r, 1].set_title(f"{name} mel spectrogram", fontsize=10)
        fig.colorbar(img, ax=axes[r, 1], format="%+2.0f dB")

    plt.tight_layout()
    out = config.FIGURES / "audio_01_waveforms_spectrograms.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def plot_augmentations(y: np.ndarray, sr: int, slug: str) -> None:
    """Draw the same clip after each of the four changes we apply to it."""
    fig, axes = plt.subplots(2, len(AUGMENTATIONS), figsize=(3.2 * len(AUGMENTATIONS), 6))
    fig.suptitle(f"Audio augmentations for {config.TEAM.get(slug, slug)}", fontsize=13, weight="bold")

    for c, (name, fn) in enumerate(AUGMENTATIONS.items()):
        ya = fn(y, sr)
        librosa.display.waveshow(ya, sr=sr, ax=axes[0, c], color="#55A868")
        axes[0, c].set_title(name, fontsize=10)
        axes[0, c].set_ylim(-1, 1)

        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=ya, sr=sr, n_mels=64), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1, c], cmap="magma")
        if c > 0:
            axes[0, c].set_ylabel("")
            axes[1, c].set_ylabel("")

    plt.tight_layout()
    out = config.FIGURES / "audio_02_augmentations.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def run():
    rows = []
    display_clips = {}

    slugs = sorted(d.name for d in config.AUDIO.iterdir() if d.is_dir())

    # Tell the user which members still have no audio so they know what to record.
    missing = [s for s in config.TEAM if not config.list_media(config.AUDIO / s, config.AUDIO_EXTS)]
    if missing:
        print("  [note] no audio yet for: " + ", ".join(missing))
        print(f"         add each member's clips to data/audio/<slug>/ (accepted: {', '.join(config.AUDIO_EXTS)})\n")

    for slug in slugs:
        for path in config.list_media(config.AUDIO / slug, config.AUDIO_EXTS):
            try:
                y, sr = librosa.load(path, sr=SR, mono=True)
            except Exception as e:
                print(f"  [warn] could not decode {path.name}: {e}")
                if path.suffix.lower() in (".m4a", ".aac", ".wma", ".opus"):
                    print(f"          {path.suffix} needs ffmpeg which is not installed. "
                          f"Convert it to .wav or .mp3 and drop that in instead.")
                continue
            if len(y) < sr * 0.5:
                print(f"  [warn] clip under 0.5s, skipping: {path.name}")
                continue

            phrase = next((k for k in config.PHRASES if k in path.stem.lower()), "unknown")
            # Keep one clip for each member for the figures and prefer the yes_approve one.
            if slug not in display_clips or phrase == "yes_approve":
                display_clips[slug] = (y, sr)

            for aug_name, fn in AUGMENTATIONS.items():
                ya = fn(y, sr)
                feats = extract_features(ya, sr)
                rows.append({
                    "member": slug,
                    "member_name": config.TEAM.get(slug, "Unauthorized / impostor"),
                    "is_authorized": int(slug in config.TEAM),
                    "phrase": phrase,
                    "phrase_text": config.PHRASES.get(phrase, "unknown"),
                    "augmentation": aug_name,
                    "source_file": path.name,
                    **feats,
                })

    if not rows:
        raise SystemExit(
            "No decodable audio found in data/audio/<member>/.\n"
            "Add each member's clips to data/audio/<slug>/ and run again."
        )

    df = pd.DataFrame(rows)
    df.to_csv(config.AUDIO_FEATURES_CSV, index=False)

    if display_clips:
        plot_waveforms_and_spectrograms(display_clips)
        first = next(iter(display_clips))
        plot_augmentations(*display_clips[first], first)
    else:
        print("  [warn] no decodable clips, skipping audio figures")

    meta = {"member", "member_name", "is_authorized", "phrase", "phrase_text",
            "augmentation", "source_file"}
    print(f"\n  identities      : {df['member'].nunique()}")
    print(f"  source clips    : {df['source_file'].nunique()}")
    print(f"  augmentations   : {len(AUGMENTATIONS) - 1} per clip (+ original)")
    print(f"  rows written    : {len(df)}")
    print(f"  features / row  : {len([c for c in df.columns if c not in meta])}")
    print(f"  -> {config.AUDIO_FEATURES_CSV}")
    return df


if __name__ == "__main__":
    run()
