"""Combined media helper for Tasks 2 and 3 by Adossi Fred William.

This file standardises every image to a .jpeg file and every voice clip to a .wav file.
"""
import subprocess

import imageio_ffmpeg
import pillow_heif
from PIL import Image

from src import config

pillow_heif.register_heif_opener()
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Any image in one of these formats is turned into a .jpeg file.
IMAGE_TO_JPEG = {".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2", ".ppm", ".heic", ".heif"}
# Any voice clip in one of these formats is turned into a .wav file.
AUDIO_TO_WAV = {".mp3", ".ogg", ".flac", ".aiff", ".aif", ".au", ".mp4", ".m4a", ".aac", ".opus", ".wma", ".3gp", ".amr"}


def convert_image(path):
    """Turn one image into a .jpeg file, renaming a .jpg directly since it is already jpeg."""
    out = path.with_suffix(".jpeg")
    if path.suffix.lower() == ".jpg":
        path.rename(out)
    else:
        Image.open(path).convert("RGB").save(out, "JPEG", quality=95)
        path.unlink()
    return out


def convert_audio(path):
    """Use ffmpeg to turn one voice clip into a mono .wav file with the same name."""
    out = path.with_suffix(".wav")
    cmd = [FFMPEG, "-y", "-i", str(path), "-ac", "1", "-ar", "44100", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] could not convert {path.name}")
        print("          " + result.stderr.strip().splitlines()[-1])
        return None
    path.unlink()
    return out


def run(verbose=True):
    """Convert every image to .jpeg and every voice clip to .wav, and return the count."""
    changed = 0

    for folder in sorted(p for p in config.IMAGES.iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_TO_JPEG:
                out = convert_image(f)
                if verbose:
                    print(f"  image  {f.name}  ->  {out.name}")
                changed += 1

    for folder in sorted(p for p in config.AUDIO.iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in AUDIO_TO_WAV:
                out = convert_audio(f)
                if out is not None:
                    if verbose:
                        print(f"  audio  {f.name}  ->  {out.name}")
                    changed += 1

    return changed


def main():
    changed = run(verbose=True)
    if changed == 0:
        print("Nothing to convert. Every file is already a .jpeg image or a .wav clip.")
    else:
        print(f"\nConverted {changed} file(s). Now run: python -m src.train_all")


if __name__ == "__main__":
    main()
