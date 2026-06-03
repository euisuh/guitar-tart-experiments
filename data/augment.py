import torch
import librosa
import numpy as np
from typing import List, Callable


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            waveform = t(waveform)
        return waveform


class GainJitter:
    def __init__(self, min_db: float = -6.0, max_db: float = 6.0):
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        db = torch.empty(1).uniform_(self.min_db, self.max_db).item()
        return waveform * (10 ** (db / 20.0))


class GaussianNoise:
    def __init__(self, snr_db: float = 30.0):
        self.snr_db = snr_db

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (self.snr_db / 10.0))
        return waveform + torch.randn_like(waveform) * noise_power.sqrt()


class PitchShift:
    def __init__(self, sample_rate: int, min_semitones: float = -2.0, max_semitones: float = 2.0):
        self.sample_rate = sample_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        n_steps = float(torch.empty(1).uniform_(self.min_semitones, self.max_semitones).item())
        x = waveform.numpy()
        shifted = librosa.effects.pitch_shift(x, sr=self.sample_rate, n_steps=n_steps)
        return torch.from_numpy(shifted)


class TimeStretch:
    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2,
                 sample_rate: int = 24000, target_length: int = 240000):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_length = target_length

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        rate = float(torch.empty(1).uniform_(self.min_rate, self.max_rate).item())
        stretched = librosa.effects.time_stretch(waveform.numpy(), rate=rate)
        stretched = torch.from_numpy(stretched)
        if stretched.shape[0] < self.target_length:
            stretched = torch.nn.functional.pad(stretched, (0, self.target_length - stretched.shape[0]))
        return stretched[:self.target_length]


def build_train_transform(sample_rate: int, segment_samples: int) -> Compose:
    return Compose([
        GainJitter(min_db=-6.0, max_db=6.0),
        GaussianNoise(snr_db=30.0),
        PitchShift(sample_rate=sample_rate, min_semitones=-2.0, max_semitones=2.0),
        TimeStretch(min_rate=0.8, max_rate=1.2, sample_rate=sample_rate, target_length=segment_samples),
    ])
