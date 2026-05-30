# Training Run V2

## Command

```powershell
python scripts/train.py --data-root archive --epochs 15 --batch-size 64 --lr 2e-4 --num-workers 4 --pretrained --view frontal --uncertain-policy one --scheduler cosine --output checkpoints/chexpert_densenet121_v2.pt
```

## Configuration

- Checkpoint: `checkpoints/chexpert_densenet121_v2.pt`
- Dataset root: `archive`
- Train CSV: `archive/train.csv`
- Valid CSV: `archive/valid.csv`
- Train rows: 191027
- Valid rows: 202
- Labels: competition 5-label preset
- View: frontal
- Uncertain policy: `one`
- Seed: 42
- AMP: enabled
- Positive class weighting: enabled
- Pretrained ImageNet DenseNet121: enabled
- Batch size: 64
- Learning rate: `2e-4`
- Scheduler: cosine
- Device: CUDA

## Best Checkpoint

The saved checkpoint corresponds to epoch 4, not epoch 15. Later epochs overfit the validation split.

Epoch 4:

- Train loss: 0.7505
- Valid loss: 0.7673
- Mean AUC: 0.8765

Independent evaluate run:

- Loss: 0.419824230788958
- Mean AUC: 0.8763721175369092

## Per-label AUC

| Label | AUC |
| --- | ---: |
| Atelectasis | 0.8424146981627297 |
| Cardiomegaly | 0.7972370766488412 |
| Consolidation | 0.8920955882352942 |
| Edema | 0.9333333333333333 |
| Pleural Effusion | 0.9167798913043478 |

## Post-evaluation Artifacts

Generated with:

```powershell
python scripts/threshold_report.py --checkpoint checkpoints/chexpert_densenet121_v2.pt --data-root archive --csv archive/valid.csv --batch-size 64 --num-workers 4 --uncertain-policy one --view frontal --output-dir outputs/evaluation
```

Artifacts:

- `outputs/evaluation/threshold_report.csv`
- `outputs/evaluation/thresholds.json`
- `outputs/evaluation/sample_predictions.csv`
