"""
Converts Guitar-TECHS raw audio files to a unified annotations.csv.

Guitar-TECHS expected directory layout after extraction:
  data/raw/guitar-techs/
    audio/
      musician1/
        technique_name/   <- folder name = technique label
          egocentric.wav
          exocentric.wav
          di.wav
          amp.wav
      musician2/
        ...

If the actual layout differs, update _discover_segments() to match.

Annotations CSV schema: audio_path, start_sec, end_sec, technique, channel
"""

import argparse
import csv
from pathlib import Path

TECHNIQUE_MAP = {
    "bend": "bend",
    "bending": "bend",
    "vibrato": "vibrato",
    "palm_mute": "palm_mute",
    "palm mute": "palm_mute",
    "pinch_harmonic": "pinch_harmonic",
    "pinch harmonic": "pinch_harmonic",
    "harmonic": "harmonic",
    "natural_harmonic": "harmonic",
    "clean": "clean",
    "normal": "clean",
    "single_note": "clean",
}

CHANNELS = ["egocentric", "exocentric", "di", "amp"]


def _discover_segments(data_dir: Path) -> list:
    import torchaudio

    records = []
    audio_root = data_dir / "audio"

    if not audio_root.exists():
        audio_root = data_dir

    for wav_path in sorted(audio_root.rglob("*.wav")):
        # Infer technique from parent directory name
        technique_raw = wav_path.parent.name.lower()
        technique = TECHNIQUE_MAP.get(technique_raw)
        if technique is None:
            technique_raw = wav_path.parent.parent.name.lower()
            technique = TECHNIQUE_MAP.get(technique_raw)
        if technique is None:
            continue

        stem = wav_path.stem.lower()
        channel = next((c for c in CHANNELS if c in stem), "unknown")

        try:
            info = torchaudio.info(str(wav_path))
            duration = info.num_frames / info.sample_rate
        except Exception:
            continue

        records.append({
            "audio_path": str(wav_path.resolve()),
            "start_sec": 0.0,
            "end_sec": round(duration, 3),
            "technique": technique,
            "channel": channel,
        })

    return records


def generate_annotations_csv(data_dir: str) -> Path:
    data_path = Path(data_dir)
    records = _discover_segments(data_path)

    if not records:
        raise RuntimeError(
            f"No annotated WAV files found in {data_dir}. "
            "Check that the dataset extracted correctly and update _discover_segments() "
            "if the directory layout differs from the expected structure."
        )

    out_path = data_path / "annotations.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "start_sec", "end_sec", "technique", "channel"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} segments to {out_path}")
    technique_counts: dict = {}
    for r in records:
        technique_counts[r["technique"]] = technique_counts.get(r["technique"], 0) + 1
    for t, n in sorted(technique_counts.items()):
        print(f"  {t}: {n}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/guitar-techs")
    args = parser.parse_args()
    generate_annotations_csv(args.data_dir)
