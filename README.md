# guitar-tart-experiments

**Status: in progress. No results yet.** Code and plan for finetuning MERT for guitar technique classification on Guitar-TECHS, with the TART paper's reported accuracy as the reference point. Nothing here has been shown to beat it.

## Plan

TART (arXiv 2510.02597) is reported to use an MLP classifier on Mel features, with 76% technique accuracy (figure taken from the paper, not reproduced here). Hypothesis to test: finetuning [MERT-v1](https://huggingface.co/m-a-p/MERT-v1-95M), a pretrained music model, does better on rare techniques than that baseline. The repo has the dataset prep, training loop, MLP baseline, and evaluation script for this; no runs are recorded yet.

## Results

| Model | Overall Acc | Macro F1 |
|---|---|---|
| TART MLP (paper, trained on IDMT) | 76% (reported in paper) | not reported here |
| MLP baseline (this repo, Guitar-TECHS) | not run yet | not run yet |
| MERT-95M (local MPS/CPU) | not run yet | not run yet |
| MERT-330M (cloud CUDA) | not run yet | not run yet |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

1. Get the Guitar-TECHS download URL from https://guitar-techs.github.io/
2. Run:
```bash
bash scripts/setup_dataset.sh <DOWNLOAD_URL>
```
This downloads to `data/raw/guitar-techs/` and generates `annotations.csv`.

## Train

**Local (Mac MPS or CPU):**
```bash
python training/train.py
```

**Cloud (CUDA, larger model):**
```bash
python training/train.py \
  --model-name m-a-p/MERT-v1-330M \
  --batch-size 32 \
  --precision fp16
```

**MLP Baseline:**
```bash
python training/train.py --model-type mlp
```

## Evaluate

```bash
python eval/evaluate.py --checkpoint checkpoints/best.pt
```

Prints accuracy, macro F1, per-class F1 vs TART baseline. Saves confusion matrix PNG.

## Tests

```bash
python -m pytest tests/ -v
```

## Cloud (Colab)

Open `notebooks/colab_train.ipynb` — mounts Google Drive, installs deps, trains both MERT-330M and MLP baseline, shows confusion matrices.

## Paper References

- **TART baseline:** [arXiv 2510.02597](https://arxiv.org/abs/2510.02597)
- **MERT:** [arXiv 2306.00107](https://arxiv.org/abs/2306.00107) | HuggingFace: `m-a-p/MERT-v1-95M` / `m-a-p/MERT-v1-330M`
- **Guitar-TECHS dataset:** https://guitar-techs.github.io/
- **SynthTab (synthetic pretraining):** [arXiv 2309.09085](https://arxiv.org/abs/2309.09085)
- **Domain adaptation:** [arXiv 2402.15258](https://arxiv.org/abs/2402.15258)

## License

MIT
