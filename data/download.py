"""
Converts Guitar-TECHS raw audio files to a unified annotations.csv.

Actual Guitar-TECHS layout (P1/P2 techniques + singlenotes):
  data/raw/guitar-techs/
    P1_techniques/audio/{directinput,micamp}/{channel}_{TechniqueName}.wav
    P1_singlenotes/audio/{directinput,micamp}/{channel}_allsinglenotes.wav
    P2_techniques/  (same structure)
    P2_singlenotes/ (same structure)

Technique is encoded in the filename stem (after the channel prefix).
Each WAV is a long continuous recording — sliced into 10s segments with 1s hop.

Annotations CSV schema: audio_path, start_sec, end_sec, technique, channel
"""

import argparse
import csv
from pathlib import Path

# Map lowercase filename keyword → canonical technique label
FILENAME_TECHNIQUE_MAP = {
    "bendings": "bend",
    "bending": "bend",
    "harmonics": "harmonic",
    "harmonic": "harmonic",
    "palmmute": "palm_mute",
    "palm_mute": "palm_mute",
    "pinchharmonics": "pinch_harmonic",
    "pinch_harmonics": "pinch_harmonic",
    "pinchharmonic": "pinch_harmonic",
    "vibrato": "vibrato",
    "allsinglenotes": "clean",
    "singlenotes": "clean",
    "clean": "clean",
}

SEGMENT_LENGTH = 10.0   # seconds, matches TART protocol
HOP_LENGTH = 1.0        # seconds


def _technique_from_stem(stem: str) -> str | None:
    """Extract technique label from filename stem like 'directinput_PalmMute'."""
    # Remove channel prefix (everything before first underscore)
    parts = stem.split("_", 1)
    name = parts[-1].lower().replace("_", "")
    return FILENAME_TECHNIQUE_MAP.get(name)


def _channel_from_folder(folder_name: str) -> str:
    folder = folder_name.lower()
    if "directinput" in folder or "di" in folder:
        return "directinput"
    if "micamp" in folder or "mic" in folder:
        return "micamp"
    return folder


def _discover_segments(data_dir: Path) -> list:
    import soundfile as sf

    records = []

    for wav_path in sorted(data_dir.rglob("*.wav")):
        if "__MACOSX" in str(wav_path):
            continue

        technique = _technique_from_stem(wav_path.stem)
        if technique is None:
            continue

        channel = _channel_from_folder(wav_path.parent.name)

        try:
            info = sf.info(str(wav_path))
            duration = info.duration
        except Exception:
            continue

        # Sliding window: 10s segments, 1s hop
        start = 0.0
        while start + SEGMENT_LENGTH <= duration:
            records.append({
                "audio_path": str(wav_path.resolve()),
                "start_sec": round(start, 3),
                "end_sec": round(start + SEGMENT_LENGTH, 3),
                "technique": technique,
                "channel": channel,
            })
            start += HOP_LENGTH

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
