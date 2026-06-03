# guitar-tart-experiments

Beating the TART baseline (76% technique accuracy) on Guitar-TECHS using MERT finetuning.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

1. Get the Guitar-TECHS download link from https://guitar-techs.github.io/
2. Run: `bash scripts/setup_dataset.sh <DOWNLOAD_URL>`

## Train

```bash
# Local (MPS/CPU)
python training/train.py

# Cloud (CUDA, larger model)
python training/train.py --model-name m-a-p/MERT-v1-330M --batch-size 32 --precision fp16
```

## Evaluate

```bash
python eval/evaluate.py --checkpoint checkpoints/best.pt
```
