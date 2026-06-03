import torch
from models.baseline_mlp import MLPBaseline
from training.config import TECHNIQUE_CLASSES

BATCH = 4
SAMPLE_RATE = 24000
SEGMENT_SAMPLES = SAMPLE_RATE * 10


def test_mlp_forward_output_shape():
    model = MLPBaseline(num_classes=len(TECHNIQUE_CLASSES), sample_rate=SAMPLE_RATE)
    x = torch.randn(BATCH, SEGMENT_SAMPLES)
    logits = model(x)
    assert logits.shape == (BATCH, len(TECHNIQUE_CLASSES))


def test_mlp_output_is_finite():
    model = MLPBaseline(num_classes=len(TECHNIQUE_CLASSES), sample_rate=SAMPLE_RATE)
    x = torch.randn(BATCH, SEGMENT_SAMPLES)
    logits = model(x)
    assert torch.isfinite(logits).all()


from unittest.mock import MagicMock, patch
from models.mert_classifier import MERTClassifier

MERT_HIDDEN = 768


def _make_mock_mert_output(batch, time_steps, hidden):
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(batch, time_steps, hidden)
    return mock_output


def test_mert_classifier_forward_shape():
    with patch("models.mert_classifier.AutoModel.from_pretrained") as mock_model_cls, \
         patch("models.mert_classifier.AutoProcessor.from_pretrained") as mock_proc_cls:
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = MERT_HIDDEN
        mock_encoder.return_value = _make_mock_mert_output(BATCH, 100, MERT_HIDDEN)
        mock_model_cls.return_value = mock_encoder
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_values": torch.randn(BATCH, SEGMENT_SAMPLES)}
        mock_proc_cls.return_value = mock_proc
        model = MERTClassifier(model_name="m-a-p/MERT-v1-95M", num_classes=len(TECHNIQUE_CLASSES), sample_rate=SAMPLE_RATE)
        x = torch.randn(BATCH, SEGMENT_SAMPLES)
        logits = model(x)
        assert logits.shape == (BATCH, len(TECHNIQUE_CLASSES))


def test_mert_classifier_freeze_encoder():
    with patch("models.mert_classifier.AutoModel.from_pretrained") as mock_model_cls, \
         patch("models.mert_classifier.AutoProcessor.from_pretrained") as mock_proc_cls:
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = MERT_HIDDEN
        mock_encoder.parameters.return_value = [torch.nn.Parameter(torch.randn(3, 3))]
        mock_model_cls.return_value = mock_encoder
        mock_proc_cls.return_value = MagicMock()
        model = MERTClassifier("m-a-p/MERT-v1-95M", len(TECHNIQUE_CLASSES), SAMPLE_RATE)
        model.freeze_encoder()
        for p in model.head.parameters():
            assert p.requires_grad


def test_mert_classifier_unfreeze_top_layers():
    with patch("models.mert_classifier.AutoModel.from_pretrained") as mock_model_cls, \
         patch("models.mert_classifier.AutoProcessor.from_pretrained") as mock_proc_cls:
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = MERT_HIDDEN
        mock_model_cls.return_value = mock_encoder
        mock_proc_cls.return_value = MagicMock()
        model = MERTClassifier("m-a-p/MERT-v1-95M", len(TECHNIQUE_CLASSES), SAMPLE_RATE)
        model.unfreeze_top_layers(n=4)  # should not raise
