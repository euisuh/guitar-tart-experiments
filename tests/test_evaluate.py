import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock
from eval.evaluate import run_eval, print_comparison_table
from training.config import TECHNIQUE_CLASSES

NUM_CLASSES = len(TECHNIQUE_CLASSES)


def _make_perfect_loader():
    labels = torch.arange(NUM_CLASSES)
    waveforms = torch.randn(NUM_CLASSES, 240000)
    return DataLoader(TensorDataset(waveforms, labels), batch_size=NUM_CLASSES)


def _make_mock_model(labels):
    model = MagicMock()
    model.eval.return_value = None
    logits = torch.zeros(len(labels), NUM_CLASSES)
    for i, lbl in enumerate(labels):
        logits[i, lbl] = 10.0
    model.return_value = logits
    return model


def test_run_eval_perfect_accuracy():
    labels = list(range(NUM_CLASSES))
    loader = _make_perfect_loader()
    model = _make_mock_model(labels)
    metrics = run_eval(model, loader, device="cpu")
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_run_eval_returns_required_keys():
    labels = list(range(NUM_CLASSES))
    loader = _make_perfect_loader()
    model = _make_mock_model(labels)
    metrics = run_eval(model, loader, device="cpu")
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class_f1" in metrics
    assert set(metrics["per_class_f1"].keys()) == set(TECHNIQUE_CLASSES)


def test_print_comparison_table_runs_without_error():
    metrics = {"accuracy": 0.87, "macro_f1": 0.83, "per_class_f1": {t: 0.80 for t in TECHNIQUE_CLASSES}}
    print_comparison_table(metrics)
