# tests/test_dataset.py
import torch
from data.dataset import GuitarTECHSDataset, TECHNIQUE_CLASSES, LABEL2IDX

SEGMENT_SAMPLES = int(10.0 * 24000)  # 10s at 24kHz


def test_dataset_length(annotations_csv):
    ds = GuitarTECHSDataset(annotations_csv, sample_rate=24000, segment_length=10.0)
    assert len(ds) == len(TECHNIQUE_CLASSES)


def test_dataset_returns_correct_shapes(annotations_csv):
    ds = GuitarTECHSDataset(annotations_csv, sample_rate=24000, segment_length=10.0)
    waveform, label, meta = ds[0]
    assert isinstance(waveform, torch.Tensor)
    assert waveform.shape == (SEGMENT_SAMPLES,)
    assert isinstance(label, int)
    assert 0 <= label < len(TECHNIQUE_CLASSES)


def test_dataset_truncates_long_audio(annotations_csv):
    # synthetic audio is 12s, segment_length=10s — should truncate
    ds = GuitarTECHSDataset(annotations_csv, sample_rate=24000, segment_length=10.0)
    waveform, _, _ = ds[0]
    assert waveform.shape[0] == SEGMENT_SAMPLES


def test_dataset_label_matches_technique(annotations_csv):
    ds = GuitarTECHSDataset(annotations_csv, sample_rate=24000, segment_length=10.0)
    for i in range(len(ds)):
        waveform, label, meta = ds[i]
        assert label == LABEL2IDX[meta["technique"]]


def test_class_weights_shape(annotations_csv):
    ds = GuitarTECHSDataset(annotations_csv, sample_rate=24000, segment_length=10.0)
    weights = ds.class_weights()
    assert weights.shape == (len(TECHNIQUE_CLASSES),)
    assert (weights > 0).all()


def test_train_val_split_sizes(annotations_csv):
    train_ds, val_ds = GuitarTECHSDataset.train_val_split(
        annotations_csv, train_ratio=0.8, seed=42, sample_rate=24000, segment_length=10.0
    )
    total = len(train_ds) + len(val_ds)
    assert total == len(TECHNIQUE_CLASSES)
    assert len(train_ds) >= 1
    assert len(val_ds) >= 1
