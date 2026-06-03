import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from training.config import TECHNIQUE_CLASSES
from data.dataset import GuitarTECHSDataset, IDX2LABEL

TART_BASELINE = {"accuracy": 0.76, "macro_f1": None, "per_class_f1": {}}


def run_eval(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            waveforms, labels = batch[0], batch[1]
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None,
                                labels=list(range(len(TECHNIQUE_CLASSES))), zero_division=0)
    per_class_f1 = {TECHNIQUE_CLASSES[i]: float(per_class_f1_arr[i]) for i in range(len(TECHNIQUE_CLASSES))}
    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1, "all_preds": all_preds, "all_labels": all_labels}


def save_confusion_matrix(all_labels, all_preds, out_path: str) -> str:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(TECHNIQUE_CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(TECHNIQUE_CLASSES)))
    ax.set_yticks(range(len(TECHNIQUE_CLASSES)))
    ax.set_xticklabels(TECHNIQUE_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(TECHNIQUE_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Technique Classification — Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def print_comparison_table(metrics: Dict) -> None:
    print("\n" + "=" * 55)
    print(f"{'Metric':<30} {'TART Baseline':>10} {'Ours':>10}")
    print("-" * 55)
    tart_acc = f"{TART_BASELINE['accuracy']:.1%}"
    our_acc = f"{metrics['accuracy']:.1%}"
    delta = metrics['accuracy'] - TART_BASELINE['accuracy']
    flag = "↑" if delta > 0 else "↓"
    print(f"{'Overall Accuracy':<30} {tart_acc:>10} {our_acc:>10} {flag}{abs(delta):.1%}")
    macro_f1_str = f"{metrics['macro_f1']:.3f}"
    print(f"{'Macro F1':<30} {'—':>10} {macro_f1_str:>10}")
    print("\nPer-class F1:")
    for cls in TECHNIQUE_CLASSES:
        tart_f1 = TART_BASELINE["per_class_f1"].get(cls, "—")
        our_f1 = f"{metrics['per_class_f1'].get(cls, 0.0):.3f}"
        print(f"  {cls:<28} {str(tart_f1):>10} {our_f1:>10}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations-csv", default="data/raw/guitar-techs/annotations.csv")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--segment-length", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confusion-matrix-out", default="confusion_matrix.png")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    from models.mert_classifier import MERTClassifier
    from models.baseline_mlp import MLPBaseline
    model_type = cfg_dict.get("model_type", "mert")
    num_classes = cfg_dict.get("num_classes", len(TECHNIQUE_CLASSES))
    if model_type == "mlp":
        model = MLPBaseline(num_classes=num_classes, sample_rate=args.sample_rate)
    else:
        model = MERTClassifier(model_name=cfg_dict.get("model_name", "m-a-p/MERT-v1-95M"),
                               num_classes=num_classes, sample_rate=args.sample_rate)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    _, val_ds = GuitarTECHSDataset.train_val_split(args.annotations_csv,
                                                    sample_rate=args.sample_rate,
                                                    segment_length=args.segment_length)

    def collate_fn(batch):
        waveforms, labels, _ = zip(*batch)
        return torch.stack(waveforms), torch.tensor(labels)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    metrics = run_eval(model, val_loader, device)
    save_confusion_matrix(metrics["all_labels"], metrics["all_preds"], args.confusion_matrix_out)
    print_comparison_table(metrics)
    print(f"Confusion matrix saved to {args.confusion_matrix_out}")


if __name__ == "__main__":
    main()
