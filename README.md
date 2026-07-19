# Formative 2 - Multimodal Data Preprocessing and Prediction Pipeline

A system that verifies a user's identity with **two biometric checks** and then reveals a
**product recommendation**. The user is verified first by their **face** and then by their
**voice**, and the recommended product is shown only once *both* checks succeed and both belong
to the *same person*. If either check fails, the user gets an **ACCESS DENIED** message and
nothing is shown.

---

## How the system works

The system follows the flow defined in the assignment:

1. The user starts at the **face check**. If the face is not recognised as a team member, or the
   match is too weak, the system halts and returns **access denied**.
2. If the face passes, the **product recommendation model** runs but its result is **hidden**.
3. The user then goes to the **voice check**. If the voice is not recognised, or is too weak, the
   system again returns **access denied**.
4. A final rule makes the system *truly multimodal*: the **voice must belong to the same person
   as the face**. This blocks an attacker holding one photo of person A and one clip of person B.
5. Only when the face passes, the voice passes, and the identities agree does the system reveal
   the **predicted product**.

---

## Team and contributions

| Member | Face / voice class | Task led |
|---|---|---|
| Thierry Alain Tresor Ibyishaka | `thierry_alain_tresor_ibyishaka` | Task 1 - data merge and EDA |
| Adossi Fred William | `adossi_fred_william` | Tasks 2 & 3 - image and audio collection & processing |
| Tresor Shingiro Nkurunziza | `tresor_shingiro_nkurunziza` | Task 4 - model creation and evaluation |
| Homere Singizwa | `homere_singizwa` | Task 6 - system demonstration |

The full written report: https://docs.google.com/document/d/163k1ausPBALIpfOF3yZy8Z52vNjznqUgUn1IhkLawnw/edit?usp=sharing.

The video demo: https://drive.google.com/file/d/1GgkkvE3Mc3q3lCqY9x8eE1sfEA2y284u/view?usp=sharing

The step-by-step notebook is in
[notebooks/Formative2_Multimodal_Pipeline.ipynb](notebooks/Formative2_Multimodal_Pipeline.ipynb),
the report figures are in [reports/figures/](reports/figures/), and the live scores are in
[reports/metrics.json](reports/metrics.json).

---

## Prerequisites

- **Python 3.11**: Other 3.1x versions usually work, but the
  pinned package versions are validated against 3.11.

Check your Python:

```bash
python --version      # should print Python 3.11.x
```

---

## Installation

```bash
# 1. clone the repository
git clone https://github.com/Tresorshingiro/Formative2_ML_Pipeline.git
cd Formative2_ML_Pipeline

# 2. create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. install the pinned dependencies
pip install -r requirements.txt

---

## Step 1 - Train the models

Everything is built by a **single command**. It standardises the media, merges the tabular data,
extracts image and audio features, and trains all three models:

```bash
python -m src.train_all
```

This runs 5 stages in order and prints a summary at the end:

| Stage | What it does | Output |
|---|---|---|
| 1. Standardise media | Converts every image → `.jpeg` and every clip → `.wav` | files in `data/images`, `data/audio` |
| 2. Merge tabular data | Cleans and joins the two CSVs, engineers features | `data/processed/merged_dataset.csv` |
| 3. Image features | Loads, augments, extracts image features | `data/processed/image_features.csv` |
| 4. Audio features | Loads, augments, extracts MFCC / spectral features | `data/processed/audio_features.csv` |
| 5. Train models | Trains and evaluates the 3 models | `models/*.joblib`, `reports/metrics.json` |

You only need to re-run this when the **media or data changes**. Once trained, the `models/*.joblib`
files are reused by the app.

Expected tail of the output:

```
  model                       accuracy        F1   log loss
  ---------------------------------------------------------
  Facial recognition             0.920     0.899      0.585
  Voiceprint verification        0.833     0.874      0.967
  Product recommendation         0.237     0.205      1.773

  Next:  python -m app.cli_app --demo
```

---

## Step 2 - Test the system (three ways)

> The app checks that the models are trained first. If you see:
> `Models not trained. Run first: python -m src.train_all`, run Step 1.

### Way 1 - Scripted demo

Runs three ready-made scenarios back to back, pressing **ENTER** between each:

```bash
python -m app.cli_app --demo
```

| Scenario | Setup | Expected result |
|---|---|---|
| 1. Authorised user | A team member's own face + voice | **ACCESS GRANTED** - product shown |
| 2. Unauthorised attempt | Unknown face + unknown voice | **ACCESS DENIED** - not a team member |
| 3. Spoof / mismatch | Valid face of member A, voice of member B | **ACCESS DENIED** - identities differ |

This single command demonstrates the full assignment requirement: a valid transaction **and** an
unauthorised attempt.

### Way 2 - Interactive menu

Choose the face, the voice, and the customer yourself from simple numbered menus:

```bash
python -m app.cli_app --interactive
```

You will be prompted to pick:
- **Whose FACE** is presented (the 4 members, plus an unauthorised person)
- **Whose VOICE** is presented (same list)
- **Which customer** to recommend for (press ENTER to accept the default, e.g: `A100`)

Mix and match to try every combination, e.g: a real face with an impostor voice.

### Way 3 - Direct paths

Point the app at exact files. Useful for testing one member or a new sample:

```bash
python -m app.cli_app \
  --face  data/images/homere_singizwa/homere_singizwa_neutral.jpeg \
  --voice data/audio/homere_singizwa/homere_singizwa_yes_approve.wav \
  --customer A100
```

| Flag | Meaning | Default |
|---|---|---|
| `--face` | Path to a face image | (required for this mode) |
| `--voice` | Path to a voice clip | (required for this mode) |
| `--customer` | Customer key for the recommendation | `A100` |

**Try an unauthorised attempt** by pointing at the `unauthorized` folder:

```bash
python -m app.cli_app \
  --face  data/images/unauthorized/unauthorized_neutral.jpeg \
  --voice data/audio/unauthorized/unauthorized_yes_approve.wav
```

Running with no arguments prints the help text:

```bash
python -m app.cli_app
```

---

## Understanding the output

For a **granted** transaction you will see, in order:

1. `STEP 1, Facial Recognition Model` → `[PASS] face recognised as <Name> (confidence X.XX)`
2. `STEP 2, Run Product Recommendation Model` → prediction computed but **held**
3. `STEP 3, Voice Validation Model` → `[PASS] voice confirmed …` and `face and voice identities agree`
4. `STEP 4, Display Predicted Product` → **ACCESS GRANTED**, the recommended category, its
   confidence, and the full probability distribution as a bar chart.

For a **denied** transaction you will see a `[FAIL]` line explaining exactly why, followed by an
**ACCESS DENIED** box. The three reasons a check can fail:

- The face/voice is **not a team member** (best match is `unauthorized`).
- The match confidence is **below the threshold** (`0.60` for both set in
  [src/config.py](src/config.py)).
- The face and voice belong to **different people**.

---

## Results and how to read them

| Model | Algorithm | Accuracy | F1 (weighted) | Log loss |
|---|---|---|---|---|
| Facial recognition | Random Forest | 0.920 | 0.899 | 0.585 |
| Voiceprint verification | Random Forest | 0.833 | 0.874 | 0.967 |
| Product recommendation | Random Forest | 0.237 | 0.205 | 1.773 |

- The two **biometric models** are evaluated with a **grouped split**: every augmented copy of an
  image or clip stays in the same fold, so a modified copy cannot leak from train into test. The
  scores are therefore measured on media the model has **never seen**, which is why the face score
  is a realistic `0.920` rather than a perfect `1.0`.
- The **product model** does not beat its ~23% baseline, and this is **expected, not a bug**. We
  tested the target statistically and found the product category is **not associated with any
  available column**. The category was assigned at
  random when the data was created, so ~23% is the true ceiling for this dataset.

---

## The media (images and audio)

Each member contributes **three face images** (neutral, smiling, surprised) and **two voice clips**
("Yes, approve" and "Confirm transaction"). The pipeline reads the **expression** and the **phrase
from the file name**, so names must follow this pattern (the slug is the folder name):

```
data/images/<slug>/<slug>_neutral.jpeg      data/audio/<slug>/<slug>_yes_approve.wav
data/images/<slug>/<slug>_smiling.jpeg      data/audio/<slug>/<slug>_confirm_transaction.wav
data/images/<slug>/<slug>_surprised.jpeg
```

You may record in almost **any common format** (HEIC photos; mp4 / m4a / aac clips). At the start of
every training run, `normalize_media` standardises every image to `.jpeg` and every clip to `.wav`
using the bundled ffmpeg — files already in the right format are left untouched, so it is safe to
re-run many times.

**To add or replace a member's media:**

```bash
# 1. delete the old files in that member's folder
# 2. drop in the new files in any accepted format, using the names above
python -m src.train_all      # standardises the media, then rebuilds and retrains everything
```

---

## Project layout

```
data/
  raw/                     customer_social_profiles.csv, customer_transactions.csv
  processed/               merged_dataset.csv, image_features.csv, audio_features.csv
  images/<slug>/           <slug>_{neutral,smiling,surprised}.jpeg
  audio/<slug>/            <slug>_{yes_approve,confirm_transaction}.wav
src/
  config.py                paths, team names, thresholds and settings
  data_merge.py            clean, join the two id styles, aggregate, merge, engineer, check
  eda.py                   the summary numbers and four labelled figures
  normalize_media.py       standardise every image to jpeg and every clip to wav
  image_pipeline.py        load, show, augment, extract, save image_features.csv
  audio_pipeline.py        load, draw, augment, extract, save audio_features.csv
  models/
    biometric.py           shared grouped trainer for the face and the voice
    face_model.py          facial recognition model
    voice_model.py         voiceprint verification model
    product_model.py       product recommendation model
  train_all.py             runs the whole pipeline in one command
app/
  cli_app.py               the command-line app that runs the login flow
models/                    trained *.joblib files (created by train_all)
notebooks/                 the step-by-step notebook
reports/figures/           the eight figures for the report
reports/metrics.json       the live scores
```

---
