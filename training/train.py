"""
Main training script for guitar technique classification.
Usage:
  /opt/homebrew/Caskroom/miniconda/base/bin/python training/train.py --help
  /opt/homebrew/Caskroom/miniconda/base/bin/python training/train.py --model-type mlp --epochs 1
"""
import os
import sys
import subprocess
from pathlib import Path

# Ensure repo root is on sys.path when running as a script (python training/train.py)
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from training.config import ExperimentConfig, parse_config, TECHNIQUE_CLASSES
from data.dataset import GuitarTECHSDataset
from data.augment import build_train_transform
from models.mert_classifier import MERTClassifier
from models.baseline_mlp import MLPBaseline
from eval.evaluate import run_eval


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_model(cfg: ExperimentConfig) -> nn.Module:
    if cfg.model_type == "mlp":
        return MLPBaseline(num_classes=cfg.num_classes, sample_rate=cfg.sample_rate)
    return MERTClassifier(
        model_name=cfg.model_name, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, dropout=cfg.dropout,
    )


def collate_fn(batch):
    waveforms, labels, _ = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for waveforms, labels in tqdm(loader, leave=False):
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(waveforms)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    cfg = parse_config()
    device = cfg.device
    print(f"Device: {device} | Model: {cfg.model_name} | Type: {cfg.model_type}")

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.run_name or None,
        config={**cfg.to_dict(), "git_hash": get_git_hash(), "device": device},
    )

    segment_samples = int(cfg.segment_length * cfg.sample_rate)
    train_transform = build_train_transform(cfg.sample_rate, segment_samples)

    train_ds, val_ds = GuitarTECHSDataset.train_val_split(
        cfg.annotations_csv, train_ratio=cfg.train_split,
        sample_rate=cfg.sample_rate, segment_length=cfg.segment_length,
        transform=train_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = build_model(cfg).to(device)
    weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler() if (cfg.precision == "fp16" and device == "cuda") else None

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        if hasattr(model, "unfreeze_top_layers") and epoch == cfg.freeze_epochs:
            model.unfreeze_top_layers(n=cfg.unfreeze_layers)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs - epoch)
            print(f"Epoch {epoch}: unfroze top {cfg.unfreeze_layers} MERT layers")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = run_eval(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss, "train/accuracy": train_acc,
            "val/accuracy": val_metrics["accuracy"], "val/macro_f1": val_metrics["macro_f1"],
            **{f"val/f1_{cls}": val_metrics["per_class_f1"].get(cls, 0.0) for cls in TECHNIQUE_CLASSES},
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": cfg.to_dict()},
                       f"{cfg.checkpoint_dir}/best.pt")

    torch.save({"epoch": cfg.epochs, "model_state": model.state_dict(), "config": cfg.to_dict()},
               f"{cfg.checkpoint_dir}/final.pt")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
