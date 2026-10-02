import os
import numpy as np
import pandas as pd
import librosa

SR = 22050

def extract_features(path):
    y, sr = librosa.load(path, sr=SR)
    f = {}

    def add(name, M):
        # mean and standard deviation over time for every feature row
        M = np.atleast_2d(M)
        for i, row in enumerate(M):
            key = f"{name}{i}" if M.shape[0] > 1 else name
            f[f"{key}_mean"] = float(row.mean())
            f[f"{key}_std"] = float(row.std())

    add("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20))          # timbre
    add("chroma", librosa.feature.chroma_stft(y=y, sr=sr))             # harmony
    add("centroid", librosa.feature.spectral_centroid(y=y, sr=sr))     # "brightness"
    add("bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr))
    add("rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr))
    add("contrast", librosa.feature.spectral_contrast(y=y, sr=sr))
    add("zcr", librosa.feature.zero_crossing_rate(y))                  # noisiness, percussion
    add("rms", librosa.feature.rms(y=y))                               # energy
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    f["tempo"] = float(np.atleast_1d(tempo)[0])                        # BPM
    return f

if __name__ == "__main__":
    rows = []
    for genre in sorted(os.listdir("data/clips")):
        for file in sorted(os.listdir(f"data/clips/{genre}")):
            feats = extract_features(f"data/clips/{genre}/{file}")
            feats["file"] = file
            feats["label"] = genre
            rows.append(feats)
            print(f"{genre}/{file} ✓")

    df = pd.DataFrame(rows)
    df.to_csv("features.csv", index=False)
    print("\nfeatures.csv saved:", df.shape)
    print(df["label"].value_counts())
