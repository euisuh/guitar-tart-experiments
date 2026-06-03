import torch
import torch.nn as nn
import librosa
import numpy as np


class MLPBaseline(nn.Module):
    """
    MLP-based technique classifier matching TART's Stage 2 architecture.
    Input: raw waveform (batch, segment_samples)
    Output: class logits (batch, num_classes)
    Uses librosa for mel spectrogram (torchaudio requires torchcodec in this env).
    """

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 24000,
        n_mels: int = 128,
        hop_length: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length

        # Compute flattened mel feature size with dummy forward
        segment_samples = sample_rate * 10
        dummy = np.zeros(segment_samples, dtype=np.float32)
        mel = librosa.feature.melspectrogram(y=dummy, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
        flat_dim = mel.shape[0] * mel.shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _compute_mel(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, samples) — process each sample with librosa
        batch_mels = []
        for i in range(x.shape[0]):
            audio = x[i].detach().cpu().numpy()
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            batch_mels.append(torch.from_numpy(mel_db))
        return torch.stack(batch_mels)  # (batch, n_mels, time_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self._compute_mel(x).to(x.device)
        return self.classifier(mel)
