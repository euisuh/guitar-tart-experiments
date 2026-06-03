import sys
from training.config import ExperimentConfig, parse_config, TECHNIQUE_CLASSES

def test_default_config_has_correct_classes():
    cfg = ExperimentConfig()
    assert cfg.num_classes == len(TECHNIQUE_CLASSES)
    assert cfg.num_classes == 6

def test_default_config_device_is_string():
    cfg = ExperimentConfig()
    assert cfg.device in ("cuda", "mps", "cpu")

def test_parse_config_overrides_batch_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--batch-size", "64"])
    cfg = parse_config()
    assert cfg.batch_size == 64

def test_config_to_dict_contains_all_fields():
    cfg = ExperimentConfig()
    d = cfg.to_dict()
    assert "model_name" in d
    assert "batch_size" in d
    assert "lr" in d
    assert "epochs" in d
