import torch
import torch.nn as nn
import librosa
import numpy as np

# Precomputed mel shape constants (must match scripts/precompute_mel.py)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SAMPLE_RATE = 24000
SEGMENT_FRAMES = int(10.0 * SAMPLE_RATE / HOP_LENGTH)  # 468


class MLPBaseline(nn.Module):
    """
    MLP-based technique classifier matching TART's Stage 2 architecture.

    Accepts precomputed mel tensors (n_mels, segment_frames) — use with MelDataset
    for fast training. Falls back to on-the-fly librosa computation for raw waveforms.

    Input: (batch, n_mels, segment_frames) from MelDataset
        OR (batch, segment_samples) from GuitarTECHSDataset (slow)
    Output: class logits (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length

        # Global average pool over time → (batch, n_mels) → MLP
        # Much lower-dimensional than flattening (128 vs 59904 features)
        self.classifier = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts (batch, n_mels, frames) or (batch, samples). Returns (batch, n_mels, frames)."""
        if x.ndim == 3:
            return x
        # on-the-fly librosa fallback
        batch_mels = []
        for i in range(x.shape[0]):
            audio = x[i].detach().cpu().numpy()
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length
            )
            batch_mels.append(torch.from_numpy(librosa.power_to_db(mel, ref=np.max)))
        return torch.stack(batch_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        mel = self._to_mel(x).to(device)          # (batch, n_mels, frames)
        pooled = mel.mean(dim=-1)                  # (batch, n_mels) — global avg pool
        return self.classifier(pooled)
