import torch
import pytest
from data.augment import Compose, PitchShift, TimeStretch, GainJitter, GaussianNoise

SAMPLE_RATE = 24000
SAMPLES = 24000 * 10


@pytest.fixture
def mono_audio():
    return torch.randn(SAMPLES)


def test_gain_jitter_preserves_shape(mono_audio):
    aug = GainJitter(min_db=-6.0, max_db=6.0)
    out = aug(mono_audio)
    assert out.shape == mono_audio.shape


def test_gaussian_noise_preserves_shape(mono_audio):
    aug = GaussianNoise(snr_db=30.0)
    out = aug(mono_audio)
    assert out.shape == mono_audio.shape


def test_gaussian_noise_changes_signal(mono_audio):
    aug = GaussianNoise(snr_db=10.0)
    out = aug(mono_audio)
    assert not torch.allclose(out, mono_audio)


def test_pitch_shift_preserves_shape(mono_audio):
    aug = PitchShift(sample_rate=SAMPLE_RATE, min_semitones=-2, max_semitones=2)
    out = aug(mono_audio)
    assert out.shape == mono_audio.shape


def test_time_stretch_preserves_shape(mono_audio):
    aug = TimeStretch(min_rate=0.9, max_rate=1.1, sample_rate=SAMPLE_RATE, target_length=SAMPLES)
    out = aug(mono_audio)
    assert out.shape == mono_audio.shape


def test_compose_applies_transforms_in_order(mono_audio):
    aug = Compose([GainJitter(min_db=-6.0, max_db=6.0), GaussianNoise(snr_db=30.0)])
    out = aug(mono_audio)
    assert out.shape == mono_audio.shape
