"""Task 6 by Homere Singizwa.

This file is the command line app that runs the face check, the voice check and the product flow.
"""
import argparse
import sys
import time
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

from src import config
from src.models import face_model, product_model, voice_model

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"


def rule(char="=", n=64):
    print(char * n)


def step(text):
    print(f"\n{BOLD}{CYAN}>> {text}{RESET}")


def ok(text):
    print(f"   {GREEN}[PASS]{RESET} {text}")


def fail(text):
    print(f"   {RED}[FAIL]{RESET} {text}")


def info(text):
    print(f"   {DIM}{text}{RESET}")


def denied(reason):
    """Print the access denied box and return False."""
    print()
    rule()
    print(f"{RED}{BOLD}  *** ACCESS DENIED ***{RESET}")
    print(f"  {reason}")
    rule()
    print()
    return False


def top_probs(probs, k=3):
    """Turn the score dictionary into a short readable line of the top few names."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(f"{name}={p:.2f}" for name, p in ranked)


def authenticate_and_recommend(face_path, voice_path, customer_key, pause=0.6):
    """Run the full flow and return True only when both the face and the voice pass."""
    rule()
    print(f"{BOLD}  MULTIMODAL TRANSACTION{RESET}")
    rule()
    info(f"face  : {Path(face_path).name}")
    info(f"voice : {Path(voice_path).name}")
    info(f"customer context: {customer_key}")
    time.sleep(pause)

    # The first gate is the face check.
    step("STEP 1, Facial Recognition Model")
    try:
        face_id, face_conf, face_probs = face_model.predict_image(face_path)
    except Exception as e:
        return denied(f"Face image could not be processed: {e}")

    info(f"top classes: {top_probs(face_probs)}")

    if face_id == config.IMPOSTOR:
        fail(f"face not recognised as a team member (best match: {config.IMPOSTOR}, {face_conf:.2f})")
        return denied("The face presented does not belong to any team member.")

    if face_conf < config.FACE_THRESHOLD:
        fail(f"matched {face_id} but confidence {face_conf:.2f} is below the threshold {config.FACE_THRESHOLD}")
        return denied(f"Face matched too weakly to allow access (confidence {face_conf:.2f}).")

    ok(f"face recognised as {BOLD}{config.TEAM[face_id]}{RESET} (confidence {face_conf:.2f})")
    time.sleep(pause)

    # We run the recommendation now but we keep it hidden until the voice passes.
    step("STEP 2, Run Product Recommendation Model")
    try:
        category, cat_conf, cat_probs = product_model.recommend_for_customer(customer_key)
    except Exception as e:
        return denied(f"Recommendation failed: {e}")

    ok("prediction computed and held until the voice is confirmed")
    info(f"{YELLOW}result is hidden until voice validation passes{RESET}")
    time.sleep(pause)

    # The second gate is the voice check.
    step("STEP 3, Voice Validation Model")
    try:
        voice_id, voice_conf, voice_probs = voice_model.predict_audio(voice_path)
    except Exception as e:
        return denied(f"Voice sample could not be processed: {e}")

    info(f"top speakers: {top_probs(voice_probs)}")

    if voice_id == config.IMPOSTOR:
        fail(f"voice not recognised as a team member (best match: {config.IMPOSTOR}, {voice_conf:.2f})")
        return denied("The voice presented does not belong to any team member.")

    if voice_conf < config.VOICE_THRESHOLD:
        fail(f"matched {voice_id} but confidence {voice_conf:.2f} is below the threshold {config.VOICE_THRESHOLD}")
        return denied(f"Voice matched too weakly to approve (confidence {voice_conf:.2f}).")

    # This check makes the system truly multimodal because the face and the voice
    # must be the same person, which stops an attacker who has one photo and one clip
    # of two different people.
    if voice_id != face_id:
        fail(f"voice belongs to {voice_id}, but the face was {face_id}")
        return denied("Face and voice belong to different people, the identities must match.")

    ok(f"voice confirmed as {BOLD}{config.TEAM[voice_id]}{RESET} (confidence {voice_conf:.2f})")
    ok("face and voice identities agree")
    time.sleep(pause)

    # Both gates passed, so now we can show the product.
    step("STEP 4, Display Predicted Product")
    print()
    rule()
    print(f"{GREEN}{BOLD}  *** ACCESS GRANTED ***{RESET}")
    print(f"  Verified user : {BOLD}{config.TEAM[face_id]}{RESET}")
    print(f"  Customer      : {customer_key}")
    print()
    print(f"  {BOLD}Recommended product category: {GREEN}{category}{RESET}  ({cat_conf:.1%} confidence)")
    print()
    print("  Full distribution:")
    for name, p in sorted(cat_probs.items(), key=lambda kv: kv[1], reverse=True):
        bar = "#" * int(p * 40)
        print(f"    {name:<12} {p:5.1%}  {bar}")
    rule()
    print()
    return True


def media_for(slug, expression="neutral", phrase="yes_approve"):
    """Find the image and the audio file for one member no matter what ending they use."""
    images = config.list_media(config.IMAGES / slug, config.IMAGE_EXTS)
    audios = config.list_media(config.AUDIO / slug, config.AUDIO_EXTS)
    image = next((p for p in images if expression in p.stem.lower()), images[0] if images else None)
    audio = next((p for p in audios if phrase in p.stem.lower()), audios[0] if audios else None)
    return image, audio


def pick_customer():
    """Read the merged dataset and return the sorted list of customer keys."""
    import pandas as pd
    df = pd.read_csv(config.MERGED_CSV)
    return sorted(df["customer_key"].unique())


def run_demo():
    """Run three ready made scenarios one after the other."""
    customers = pick_customer()
    member = next(iter(config.TEAM))

    print(f"\n{BOLD}#### SCENARIO 1, AUTHORISED USER, the normal path ####{RESET}\n")
    face, voice = media_for(member)
    authenticate_and_recommend(face, voice, customers[0])

    input(f"{DIM}  press ENTER for the unauthorised scenario{RESET}")

    print(f"\n{BOLD}#### SCENARIO 2, UNAUTHORISED ATTEMPT, unknown face and voice ####{RESET}\n")
    face, voice = media_for(config.IMPOSTOR)
    authenticate_and_recommend(face, voice, customers[0])

    input(f"{DIM}  press ENTER for the mismatched identity scenario{RESET}")

    members = list(config.TEAM)
    print(f"\n{BOLD}#### SCENARIO 3, VALID FACE BUT WRONG VOICE, a spoof attempt ####{RESET}\n")
    print(f"{DIM}  A valid team member's face, but a different member speaks the approval.{RESET}\n")
    face, _ = media_for(members[0])
    _, voice = media_for(members[1])
    authenticate_and_recommend(face, voice, customers[0])


def run_interactive():
    """Let the user choose the face, the voice and the customer from simple menus."""
    members = list(config.TEAM) + [config.IMPOSTOR]
    customers = pick_customer()

    print(f"\n{BOLD}Whose FACE is presented?{RESET}")
    for i, m in enumerate(members, 1):
        label = config.TEAM.get(m, "Unknown person (unauthorised)")
        print(f"  {i}. {label}")
    f_idx = int(input("  > ")) - 1

    print(f"\n{BOLD}Whose VOICE is presented?{RESET}")
    for i, m in enumerate(members, 1):
        label = config.TEAM.get(m, "Unknown person (unauthorised)")
        print(f"  {i}. {label}")
    v_idx = int(input("  > ")) - 1

    print(f"\n{BOLD}Customer to recommend for?{RESET} (blank uses {customers[0]})")
    print(f"  {DIM}{', '.join(customers[:14])} and more{RESET}")
    ck = input("  > ").strip().upper() or customers[0]
    if ck not in customers:
        print(f"{RED}unknown customer {ck}{RESET}")
        return

    face, _ = media_for(members[f_idx])
    _, voice = media_for(members[v_idx])
    print()
    authenticate_and_recommend(face, voice, ck)


def main():
    ap = argparse.ArgumentParser(description="Multimodal login and product recommendation")
    ap.add_argument("--demo", action="store_true", help="run the scripted demo scenarios")
    ap.add_argument("--interactive", action="store_true", help="choose identities from a menu")
    ap.add_argument("--face", help="path to a face image")
    ap.add_argument("--voice", help="path to a voice .wav")
    ap.add_argument("--customer", default="A100", help="customer key, e.g. A100")
    args = ap.parse_args()

    for p in (config.MODELS / "face_model.joblib", config.MODELS / "voice_model.joblib",
              config.MODELS / "product_model.joblib"):
        if not p.exists():
            print(f"{RED}Models not trained. Run first:{RESET}  python -m src.train_all")
            sys.exit(1)

    if args.demo:
        run_demo()
    elif args.interactive:
        run_interactive()
    elif args.face and args.voice:
        authenticate_and_recommend(args.face, args.voice, args.customer.upper())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
