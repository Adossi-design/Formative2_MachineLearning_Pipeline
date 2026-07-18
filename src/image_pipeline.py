"""Task 2 by Adossi Fred William.

This file loads the face images, augments them, and saves their features to image_features.csv.
"""
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

RNG = np.random.default_rng(config.RANDOM_STATE)

# HOG on a 64 by 64 window that gives 144 numbers, kept small because we have few images.
HOG = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(32, 32), _blockStride=(32, 32),
    _cellSize=(16, 16), _nbins=9,
)


# The functions below are the four ways we change an image.
def aug_rotate(img, angle=15):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def aug_flip(img):
    return cv2.flip(img, 1)


def aug_grayscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def aug_bright_noise(img, gain=1.35, sigma=12):
    out = img.astype(np.float32) * gain
    out += RNG.normal(0, sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


AUGMENTATIONS = {
    "original": lambda im: im,
    "rotated": aug_rotate,
    "flipped": aug_flip,
    "grayscale": aug_grayscale,
    "bright_noise": aug_bright_noise,
}


def extract_features(img: np.ndarray) -> dict:
    """Read the colour histogram, a small pixel version, the HOG shape and some totals."""
    img = cv2.resize(img, config.FACE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = {}

    # Colour histogram with 16 bins per channel, scaled so the image size does not matter.
    for i, ch in enumerate(("b", "g", "r")):
        hist = cv2.calcHist([img], [i], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        for j, v in enumerate(hist):
            feats[f"hist_{ch}_{j:02d}"] = float(v)

    # A small 16 by 16 grey version of the face used as a simple cheap descriptor.
    emb = cv2.resize(gray, (16, 16)).flatten().astype(np.float32) / 255.0
    for j, v in enumerate(emb):
        feats[f"emb_{j:03d}"] = float(v)

    # HOG describes the shape of the face like the eyes, brows and mouth edges.
    hog = HOG.compute(cv2.resize(gray, (64, 64))).flatten()
    for j, v in enumerate(hog):
        feats[f"hog_{j:03d}"] = float(v)

    # A few overall numbers about the whole image.
    feats["mean_intensity"] = float(gray.mean())
    feats["std_intensity"] = float(gray.std())
    feats["mean_b"], feats["mean_g"], feats["mean_r"] = [float(img[:, :, i].mean()) for i in range(3)]
    edges = cv2.Canny(gray, 100, 200)
    feats["edge_density"] = float((edges > 0).mean())
    feats["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # how sharp the image is
    return feats


def display_samples(records_by_member: dict) -> None:
    """Show one image for each expression for each member so we can see the data."""
    members = list(records_by_member)
    fig, axes = plt.subplots(len(members), len(config.EXPRESSIONS),
                             figsize=(3.1 * len(config.EXPRESSIONS), 3.1 * len(members)))
    axes = np.atleast_2d(axes)
    fig.suptitle("Sample facial images, one per expression per member", fontsize=14, weight="bold")

    for r, slug in enumerate(members):
        for c, expr in enumerate(config.EXPRESSIONS):
            ax = axes[r, c]
            img = records_by_member[slug].get(expr)
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(expr, fontsize=11)
            if c == 0:
                label = config.TEAM.get(slug, slug.replace("_", " ").title())
                ax.set_ylabel(label.replace(" ", "\n", 1), fontsize=8)

    plt.tight_layout()
    out = config.FIGURES / "images_01_samples.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def display_augmentations(img: np.ndarray, slug: str) -> None:
    """Show the same face after each of the four changes we apply to it."""
    fig, axes = plt.subplots(1, len(AUGMENTATIONS), figsize=(3.0 * len(AUGMENTATIONS), 3.4))
    fig.suptitle(f"Augmentations applied to every image for {config.TEAM.get(slug, slug)}",
                 fontsize=13, weight="bold")
    for ax, (name, fn) in zip(axes, AUGMENTATIONS.items()):
        ax.imshow(cv2.cvtColor(fn(img), cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    out = config.FIGURES / "images_02_augmentations.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  saved {out}")


def run():
    rows = []
    samples = {}

    slugs = sorted(d.name for d in config.IMAGES.iterdir() if d.is_dir())

    # Tell the user which members still have no images so they know what to record.
    missing = [s for s in config.TEAM if not config.list_media(config.IMAGES / s, config.IMAGE_EXTS)]
    if missing:
        print("  [note] no images yet for: " + ", ".join(missing))
        print(f"         add each member's images to data/images/<slug>/ (accepted: {', '.join(config.IMAGE_EXTS)})\n")

    for slug in slugs:
        samples.setdefault(slug, {})
        for path in config.list_media(config.IMAGES / slug, config.IMAGE_EXTS):
            img = cv2.imread(str(path))
            if img is None:
                print(f"  [warn] unreadable image, skipping: {path.name} "
                      f"(is it a valid {path.suffix} file?)")
                continue

            expression = next((e for e in config.EXPRESSIONS if e in path.stem.lower()), "unknown")
            samples[slug].setdefault(expression, img)

            for aug_name, fn in AUGMENTATIONS.items():
                feats = extract_features(fn(img))
                rows.append({
                    "member": slug,
                    "member_name": config.TEAM.get(slug, "Unauthorized / impostor"),
                    "is_authorized": int(slug in config.TEAM),
                    "expression": expression,
                    "augmentation": aug_name,
                    "source_file": path.name,
                    **feats,
                })

    if not rows:
        raise SystemExit(
            "No readable images found in data/images/<member>/.\n"
            "Add each member's images to data/images/<slug>/ and run again."
        )

    df = pd.DataFrame(rows)
    df.to_csv(config.IMAGE_FEATURES_CSV, index=False)

    display_samples(samples)
    # Draw the augmentation figure using the neutral image, or any image if neutral is missing.
    for slug, by_expr in samples.items():
        if by_expr:
            img = by_expr[config.EXPRESSIONS[0]] if config.EXPRESSIONS[0] in by_expr \
                else next(iter(by_expr.values()))
            display_augmentations(img, slug)
            break

    n_feat = len([c for c in df.columns if c.startswith(("hist_", "emb_", "hog_")) or c in
                  ("mean_intensity", "std_intensity", "mean_b", "mean_g", "mean_r",
                   "edge_density", "laplacian_var")])
    print(f"\n  identities      : {df['member'].nunique()}")
    print(f"  source images   : {df['source_file'].nunique()}")
    print(f"  augmentations   : {len(AUGMENTATIONS) - 1} per image (+ original)")
    print(f"  rows written    : {len(df)}")
    print(f"  features / row  : {n_feat}")
    print(f"  -> {config.IMAGE_FEATURES_CSV}")
    return df


if __name__ == "__main__":
    run()
