import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.io.wavfile
from typing import Optional, Tuple, Dict, List, Callable

TECHNIQUE_CLASSES = ["bend", "harmonic", "palm_mute", "pinch_harmonic", "vibrato", "clean"]
LABEL2IDX: Dict[str, int] = {t: i for i, t in enumerate(TECHNIQUE_CLASSES)}
IDX2LABEL: Dict[int, str] = {i: t for t, i in LABEL2IDX.items()}


class GuitarTECHSDataset(Dataset):
    """
    PyTorch Dataset for Guitar-TECHS technique classification.
    CSV schema: audio_path, start_sec, end_sec, technique, channel
    Returns: (waveform, label, metadata) where waveform shape is (segment_samples,)
    """

    def __init__(
        self,
        annotations_csv: str,
        sample_rate: int = 24000,
        segment_length: float = 10.0,
        transform: Optional[Callable] = None,
        row_indices: Optional[List[int]] = None,
    ):
        df = pd.read_csv(annotations_csv)
        df = df[df["technique"].isin(TECHNIQUE_CLASSES)].reset_index(drop=True)
        if row_indices is not None:
            df = df.iloc[row_indices].reset_index(drop=True)
        self.df = df
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        technique = row["technique"]
        label = LABEL2IDX[technique]

        waveform = self._load_segment(
            str(row["audio_path"]),
            float(row["start_sec"]),
            float(row["end_sec"]),
        )

        if self.transform is not None:
            waveform = self.transform(waveform)

        metadata = {
            "audio_path": str(row["audio_path"]),
            "technique": technique,
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
        return waveform, label, metadata

    def _load_segment(self, audio_path: str, start_sec: float, end_sec: float) -> torch.Tensor:
        sr, data = scipy.io.wavfile.read(audio_path)

        # Convert to float32 in [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        # Handle stereo — average channels
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Slice to requested segment
        frame_offset = int(start_sec * sr)
        num_frames = int((end_sec - start_sec) * sr)
        data = data[frame_offset: frame_offset + num_frames]

        waveform = torch.from_numpy(data)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = waveform.unsqueeze(0)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=int(len(waveform) * self.sample_rate / sr),
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        # Truncate or pad to segment_samples
        length = waveform.shape[0]
        if length < self.segment_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_samples - length))
        else:
            waveform = waveform[:self.segment_samples]

        return waveform  # (segment_samples,)

    def class_weights(self) -> torch.Tensor:
        counts = self.df["technique"].value_counts()
        total = len(self.df)
        weights = torch.ones(len(TECHNIQUE_CLASSES))
        for i, cls_name in enumerate(TECHNIQUE_CLASSES):
            n = counts.get(cls_name, 1)
            weights[i] = total / (len(TECHNIQUE_CLASSES) * n)
        return weights

    @classmethod
    def train_val_split(
        cls,
        annotations_csv: str,
        train_ratio: float = 0.8,
        seed: int = 42,
        **kwargs,
    ) -> Tuple["GuitarTECHSDataset", "GuitarTECHSDataset"]:
        df = pd.read_csv(annotations_csv)
        df = df[df["technique"].isin(TECHNIQUE_CLASSES)].reset_index(drop=True)

        rng = np.random.RandomState(seed)
        all_indices = df.index.tolist()
        rng.shuffle(all_indices)

        # Try stratified split; fall back to global split if any class has too few samples
        train_indices: List[int] = []
        val_indices: List[int] = []

        for technique in TECHNIQUE_CLASSES:
            idxs = df.index[df["technique"] == technique].tolist()
            rng.shuffle(idxs)
            if len(idxs) >= 2:
                n_train = max(1, int(len(idxs) * train_ratio))
                train_indices.extend(idxs[:n_train])
                val_indices.extend(idxs[n_train:])
            else:
                # Only 1 sample for this class — assign to train, handle below
                train_indices.extend(idxs)

        # Guarantee at least 1 val sample: move last train item to val if val is empty
        if len(val_indices) == 0 and len(train_indices) >= 2:
            val_indices.append(train_indices.pop())

        train_ds = cls(annotations_csv, row_indices=train_indices, **kwargs)
        val_ds = cls(annotations_csv, row_indices=val_indices, **kwargs)
        return train_ds, val_ds
