import os
import sys
import numpy as np
import av
import soundfile as sf

SR = 22050      # sample rate
DUR = 30        # clip length in seconds


def load_audio(path):
    """Decode any audio file (mp3, m4a, wav, ...) into a mono 22050 Hz signal."""
    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks)


def cut_one(src, dst_dir="data/test_clips"):
    """Cut a 30-second clip from the middle of one song and save it as .wav in dst_dir. Returns the clip path."""
    y = load_audio(src)
    total = len(y) / SR
    if total < DUR:
        print(f"   ⚠ {src} is only {total:.0f} s, shorter than 30 s!")
    start = max(0, len(y) // 2 - SR * DUR // 2)        # from the middle
    clip = y[start:start + SR * DUR]
    clip = np.pad(clip, (0, SR * DUR - len(clip)))     # exactly 30 s

    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(dst_dir, name + ".wav")
    sf.write(dst, clip, SR)
    print(f"✓ {os.path.basename(src)}  ({total:.0f} s → 30 s) → {dst}")
    return dst


def cut_folder(src_dir, dst_dir):
    """Cut every song in a folder with cut_one."""
    bad = []
    for f in sorted(os.listdir(src_dir)):
        try:
            cut_one(os.path.join(src_dir, f), dst_dir)
        except Exception as e:
            bad.append(f"{src_dir}/{f}")
            print(f"✗ {src_dir}/{f}  FAILED TO READ: {e}")
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # python cut_clips.py "song.mp3"
        for p in sys.argv[1:]:
            cut_one(p)
    else:
        # python cut_clips.py
        bad = []
        for genre in sorted(os.listdir("data/raw")):
            bad += cut_folder(f"data/raw/{genre}", f"data/clips/{genre}")
        bad += cut_folder("data/test_raw", "data/test_clips")

        print("\n=== Summary ===")
        for genre in sorted(os.listdir("data/clips")):
            print(f"{genre}: {len(os.listdir(f'data/clips/{genre}'))} clips")
        print(f"test: {len(os.listdir('data/test_clips'))} clips")
        if bad:
            print("\nThese files are corrupted, replace them:")
            for b in bad:
                print("  -", b)
