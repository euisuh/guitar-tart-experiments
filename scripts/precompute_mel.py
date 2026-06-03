"""
Precomputes mel spectrograms for all unique WAV files in Guitar-TECHS.
Saves mel arrays to data/raw/guitar-techs/mel_cache/ as .npy files.
Writes mel_annotations.csv with frame-level indices instead of time offsets.

Run once before MLP training:
  python scripts/precompute_mel.py
"""
import argparse
import csv
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


SAMPLE_RATE = 24000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_LENGTH = 10.0  # seconds


def frames_per_second(sr: int, hop: int) -> float:
    return sr / hop


def compute_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute log-mel spectrogram, resampling to SAMPLE_RATE first."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # (n_mels, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", default="data/raw/guitar-techs/annotations.csv")
    parser.add_argument("--cache-dir", default="data/raw/guitar-techs/mel_cache")
    parser.add_argument("--out-csv", default="data/raw/guitar-techs/mel_annotations.csv")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Read annotations
    with open(args.annotations_csv) as f:
        rows = list(csv.DictReader(f))

    # Collect unique WAV paths
    unique_wavs = sorted({r["audio_path"] for r in rows})
    print(f"Precomputing mel for {len(unique_wavs)} unique WAV files...")

    fps = frames_per_second(SAMPLE_RATE, HOP_LENGTH)
    segment_frames = int(SEGMENT_LENGTH * fps)

    def _musician_prefix(wav_path: str) -> str:
        """Extract P1/P2 musician ID from the path hierarchy."""
        for part in Path(wav_path).parts:
            if part.startswith("P1"):
                return "P1"
            if part.startswith("P2"):
                return "P2"
        return "PX"

    def _unique_stem(wav_path: str) -> str:
        """Make a unique stem that includes musician prefix to avoid collisions."""
        prefix = _musician_prefix(wav_path)
        return f"{prefix}_{Path(wav_path).stem}"

    # Precompute mel for each unique WAV (with musician-prefixed filenames)
    mel_cache: dict[str, np.ndarray] = {}
    for i, wav_path in enumerate(unique_wavs):
        stem = _unique_stem(wav_path)
        npy_path = cache_dir / f"{stem}.npy"

        if npy_path.exists():
            print(f"  [{i+1}/{len(unique_wavs)}] {stem} — cached")
            mel_cache[wav_path] = np.load(str(npy_path), mmap_mode="r")
        else:
            print(f"  [{i+1}/{len(unique_wavs)}] {stem} — computing...")
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            mel = compute_mel(audio, sr)
            np.save(str(npy_path), mel)
            mel_cache[wav_path] = mel
            print(f"    shape: {mel.shape} → {npy_path.name}")

    # Write mel_annotations.csv with frame indices + musician column
    print(f"\nWriting {args.out_csv}...")
    out_rows = []
    skipped = 0
    for row in rows:
        wav_path = row["audio_path"]
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        mel = mel_cache[wav_path]
        total_frames = mel.shape[1]

        start_frame = int(start_sec * fps)
        end_frame = start_frame + segment_frames

        if end_frame > total_frames:
            skipped += 1
            continue

        npy_path = cache_dir / f"{_unique_stem(wav_path)}.npy"
        out_rows.append({
            "mel_path": str(npy_path.resolve()),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "technique": row["technique"],
            "channel": row["channel"],
            "musician": _musician_prefix(wav_path),
        })

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mel_path", "start_frame", "end_frame", "technique", "channel", "musician"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} segments ({skipped} skipped at file boundary)")
    technique_counts: dict = {}
    for r in out_rows:
        technique_counts[r["technique"]] = technique_counts.get(r["technique"], 0) + 1
    for t, n in sorted(technique_counts.items()):
        print(f"  {t}: {n}")
    musician_counts: dict = {}
    for r in out_rows:
        musician_counts[r["musician"]] = musician_counts.get(r["musician"], 0) + 1
    for m, n in sorted(musician_counts.items()):
        print(f"  {m}: {n} segments")


if __name__ == "__main__":
    main()
