from dataclasses import dataclass
import argparse

TECHNIQUE_CLASSES = ["bend", "harmonic", "palm_mute", "pinch_harmonic", "vibrato", "clean"]


@dataclass
class ExperimentConfig:
    # Model
    model_name: str = "m-a-p/MERT-v1-95M"
    model_type: str = "mert"
    num_classes: int = len(TECHNIQUE_CLASSES)
    dropout: float = 0.1
    freeze_epochs: int = 2
    unfreeze_layers: int = 4

    # Data
    data_dir: str = "data/raw/guitar-techs"
    annotations_csv: str = "data/raw/guitar-techs/annotations.csv"
    sample_rate: int = 24000
    segment_length: float = 10.0
    train_split: float = 0.8
    num_workers: int = 0

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 20
    precision: str = "fp32"

    # Logging
    wandb_project: str = "guitar-tart"
    wandb_entity: str = ""
    run_name: str = ""

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    @property
    def device(self) -> str:
        import platform
        # CPU device detection without torch (avoids macOS OpenMP crash in tests)
        if platform.system() == "Darwin":
            return "mps"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Guitar technique classification")
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--model-type", default="mert", choices=["mert", "mlp"])
    parser.add_argument("--num-classes", type=int, default=len(TECHNIQUE_CLASSES))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--data-dir", default="data/raw/guitar-techs")
    parser.add_argument("--annotations-csv", default="data/raw/guitar-techs/annotations.csv")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--segment-length", type=float, default=10.0)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--wandb-project", default="guitar-tart")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    return ExperimentConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        num_classes=args.num_classes,
        dropout=args.dropout,
        freeze_epochs=args.freeze_epochs,
        unfreeze_layers=args.unfreeze_layers,
        data_dir=args.data_dir,
        annotations_csv=args.annotations_csv,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        train_split=args.train_split,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        precision=args.precision,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
    )
