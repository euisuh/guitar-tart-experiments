"""
Fast dataset that loads precomputed mel spectrograms from .npy files.
Use after running: python scripts/precompute_mel.py

Returns (mel_tensor, label, metadata) where mel_tensor shape is (n_mels, segment_frames).
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

from data.dataset import TECHNIQUE_CLASSES, LABEL2IDX, IDX2LABEL


class MelDataset(Dataset):
    """
    Loads precomputed mel spectrograms from .npy files.
    CSV schema: mel_path, start_frame, end_frame, technique, channel
    """

    def __init__(
        self,
        mel_annotations_csv: str,
        row_indices: Optional[List[int]] = None,
    ):
        df = pd.read_csv(mel_annotations_csv)
        df = df[df["technique"].isin(TECHNIQUE_CLASSES)].reset_index(drop=True)
        if row_indices is not None:
            df = df.iloc[row_indices].reset_index(drop=True)
        self.df = df

        # Memory-map all unique .npy files for fast slicing
        self._mel_cache: Dict[str, np.ndarray] = {}
        for path in self.df["mel_path"].unique():
            self._mel_cache[path] = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        technique = row["technique"]
        label = LABEL2IDX[technique]

        mel_arr = self._mel_cache[row["mel_path"]]
        start, end = int(row["start_frame"]), int(row["end_frame"])
        mel = torch.from_numpy(mel_arr[:, start:end].copy())  # (n_mels, frames)

        metadata = {
            "mel_path": row["mel_path"],
            "technique": technique,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
        }
        return mel, label, metadata

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
        mel_annotations_csv: str,
        train_ratio: float = 0.8,
        seed: int = 42,
        split_by_musician: bool = True,
    ) -> Tuple["MelDataset", "MelDataset"]:
        """
        split_by_musician=True (default): P1 files → train, P2 files → val.
            Avoids data leakage from overlapping sliding-window segments.
        split_by_musician=False: random stratified split (leaky — segments from
            same recording appear in both train and val).
        """
        df = pd.read_csv(mel_annotations_csv)
        df = df[df["technique"].isin(TECHNIQUE_CLASSES)].reset_index(drop=True)

        if split_by_musician:
            # Use musician column if present, else infer from mel_path filename prefix
            if "musician" in df.columns:
                musician = df["musician"]
            else:
                def _infer_musician(path: str) -> str:
                    stem = Path(path).stem  # e.g. "P1_directinput_Bendings"
                    if stem.startswith("P1"):
                        return "P1"
                    if stem.startswith("P2"):
                        return "P2"
                    return "unknown"
                musician = df["mel_path"].apply(_infer_musician)
            train_indices = df.index[musician == "P1"].tolist()
            val_indices = df.index[musician == "P2"].tolist()
        else:
            rng = np.random.RandomState(seed)
            train_indices = []
            val_indices = []
            for technique in TECHNIQUE_CLASSES:
                idxs = df.index[df["technique"] == technique].tolist()
                rng.shuffle(idxs)
                n_train = max(1, int(len(idxs) * train_ratio))
                train_indices.extend(idxs[:n_train])
                val_indices.extend(idxs[n_train:] if len(idxs) > 1 else [])
            if not val_indices and len(train_indices) >= 2:
                val_indices.append(train_indices.pop())

        return (
            cls(mel_annotations_csv, row_indices=train_indices),
            cls(mel_annotations_csv, row_indices=val_indices),
        )
