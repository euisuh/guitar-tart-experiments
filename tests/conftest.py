# tests/conftest.py
import pytest
import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path

SAMPLE_RATE = 24000
DURATION_SEC = 12.0  # longer than 10s segment to test truncation
TECHNIQUE_CLASSES = ["bend", "harmonic", "palm_mute", "pinch_harmonic", "vibrato", "clean"]


@pytest.fixture
def tmp_audio_dir(tmp_path):
    """Create synthetic WAV files for each technique class."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    paths = {}
    for technique in TECHNIQUE_CLASSES:
        wav_path = audio_dir / f"{technique}.wav"
        n_samples = int(DURATION_SEC * SAMPLE_RATE)
        freq = 220.0 * (1 + TECHNIQUE_CLASSES.index(technique) * 0.1)
        t = np.linspace(0, DURATION_SEC, n_samples, dtype=np.float32)
        waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(wav_path), waveform, SAMPLE_RATE)
        paths[technique] = str(wav_path)
    return paths


@pytest.fixture
def annotations_csv(tmp_path, tmp_audio_dir):
    """Write a minimal annotations CSV using the synthetic audio files."""
    rows = []
    for technique, path in tmp_audio_dir.items():
        rows.append({
            "audio_path": path,
            "start_sec": 0.0,
            "end_sec": DURATION_SEC,
            "technique": technique,
            "channel": "di",
        })
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)
