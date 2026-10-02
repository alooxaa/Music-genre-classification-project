import os
import sys
import numpy as np
import pandas as pd
import joblib
from extract_features import extract_features
from cut_clips import cut_one

saved = joblib.load("genre_model.joblib")
model, columns = saved["model"], saved["columns"]


def test_one(path):
    """Classify one song: if it is not a .wav clip, cut a 30 s clip first, then predict the genre."""
    if not path.lower().endswith(".wav"):
        path = cut_one(path, "data/test_clips")

    x = pd.DataFrame([extract_features(path)])[columns]
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    genre = model.classes_[order[0]]

    print(f"🎵 {os.path.basename(path)}")
    print(f"   Genre: {genre.upper()}  ({proba[order[0]]:.0%})")
    for i in order:
        print(f"   {model.classes_[i]:12s} {proba[i]:5.0%} {'█' * int(proba[i] * 30)}")
    return genre


def clean(p):
    # strip the & ' " characters PowerShell adds when a file is dragged into the terminal
    p = p.strip()
    if p.startswith("& "):
        p = p[2:]
    return p.strip().strip('"').strip("'")


if __name__ == "__main__":
    print("Model:", saved["name"], "| genres:", ", ".join(model.classes_))

    if len(sys.argv) > 1:
        # python test.py "song.mp3"  → classify only these songs
        for p in sys.argv[1:]:
            test_one(p)
    else:
        # python test.py  → interactive mode, enter songs one by one
        while True:
            p = clean(input("\nEnter a song path or drag the file here (Enter to quit): "))
            if not p:
                break
            if not os.path.isfile(p):
                print("   ✗ File not found:", p)
                continue
            try:
                test_one(p)
            except Exception as e:
                print("   ✗ Error:", e)
