# CheXpert Web Reader

Project scaffold for a chest X-ray web app backed by a CheXpert-trained multi-label model.

## Target

Build a web application where a user uploads a chest X-ray image and receives model probabilities for CheXpert-style findings.

This is not a medical device. The app should be treated as research/demo software until the model, data handling, calibration, validation, and regulatory requirements are handled properly.

## Expected Dataset Layout

Download the CheXpert dataset from Kaggle: [ashery/chexpert](https://www.kaggle.com/datasets/ashery/chexpert).

After extraction, the current workspace expects the Kaggle-style layout under `archive/`:

```text
archive/
  train.csv
  valid.csv
  train/
    patient00001/
      study1/
        view1_frontal.jpg
  valid/
```

The loader also supports the Stanford-style CSV paths that start with `CheXpert-v1.0-small/`.

If you prefer a separate data directory, place CheXpert under `data/chexpert/`:

```text
data/
  chexpert/
    train.csv
    valid.csv
    CheXpert-v1.0-small/
      train/
      valid/
```

The loader accepts CSV rows with a `Path` column and CheXpert finding columns. If no `valid.csv` exists, the training script splits validation rows from `train.csv`.

## Quick Start

```powershell
conda activate dat
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

Without a checkpoint, the API returns image metadata and a clear `model_not_loaded` status instead of pretending to diagnose.
With a checkpoint, the API returns probabilities, a short generated report, and a Grad-CAM heatmap for the highest-probability finding.

## Train

Once the dataset exists:

```powershell
python scripts/train.py --data-root archive --epochs 5 --batch-size 32 --num-workers 4 --pretrained --view frontal --output checkpoints/chexpert_densenet121.pt
```

Add `--pretrained` if you want ImageNet initialization and the machine can download/cache torchvision weights.
Add `--label-preset all` to train all 14 CheXpert labels instead of the 5 competition labels.
Training defaults to frontal-only images, mixed precision on CUDA, fixed seed `42`, and class imbalance `pos_weight`.
The training loop shows progress bars with current batch, speed, loss, and ETA so long runs do not look frozen.
Each run writes:

- best checkpoint: `checkpoints/chexpert_densenet121.pt`
- last checkpoint: `checkpoints/chexpert_densenet121_last.pt`
- metrics/config JSON: `checkpoints/chexpert_densenet121.metrics.json`

CUDA smoke test before full training:

```powershell
python scripts/train.py --data-root archive --epochs 1 --batch-size 32 --num-workers 4 --limit 2048 --pretrained --view frontal --output checkpoints/smoke_cuda.pt
```

## Evaluate

```powershell
python scripts/evaluate.py --checkpoint checkpoints/chexpert_densenet121.pt --data-root archive --batch-size 32 --num-workers 4
```

If `archive/valid.csv` is missing, evaluation falls back to `archive/train.csv`. Use `--limit` for a quick smoke run.

## CLI Prediction

```powershell
python scripts/predict.py --checkpoint checkpoints/chexpert_densenet121.pt --image archive/train/patient00001/study1/view1_frontal.jpg
```

## Run With A Checkpoint

```powershell
conda activate dat
$env:CHEXPERT_CHECKPOINT="checkpoints/chexpert_densenet121.pt"
uvicorn app.main:app --reload
```

The prediction endpoint is `POST /api/predict`. It includes heatmaps by default; use `/api/predict?include_heatmap=false` when you only need probabilities.

## Labels

The default output labels are:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion

These are the common CheXpert competition labels. Use `--label-preset all` during training to output all 14 CheXpert labels; the web app reads the label list from the checkpoint.

## Test

```powershell
conda activate dat
python -m unittest discover
python scripts/train.py --data-root archive --epochs 0 --batch-size 2 --limit 8
```
