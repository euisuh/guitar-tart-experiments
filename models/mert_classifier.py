import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from typing import Optional


class MERTClassifier(nn.Module):
    """
    Guitar technique classifier using MERT encoder.
    Freeze encoder for freeze_epochs, then unfreeze last N transformer layers.
    Mean pool over time → linear classification head.
    """

    def __init__(self, model_name: str, num_classes: int, sample_rate: int = 24000, dropout: float = 0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, n: int = 4) -> None:
        layers = None
        for attr in ["layers", "encoder.layers", "model.layers"]:
            obj = self.encoder
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                layers = obj
                break
        if layers is None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            return
        for layer in list(layers)[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_np = x.cpu().numpy()
        inputs = self.processor(
            list(x_np), sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs["input_values"].to(device)
        outputs = self.encoder(input_values=input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.head(hidden)
