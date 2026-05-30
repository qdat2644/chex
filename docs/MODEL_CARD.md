# CheXpert Reader Model Card

## Model

- Architecture: DenseNet121
- Checkpoint: `checkpoints/chexpert_densenet121_v2.pt`
- Label preset: CheXpert competition labels
- View: frontal
- Input preprocessing: resize to 224 x 224, grayscale to 3 channels, ImageNet normalization

## Labels

The inference label order is:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Pleural Effusion

## Validation

Validation split:

- CSV: `archive/valid.csv`
- Rows: 202
- View: frontal
- Uncertain policy: `one`
- Loss: 0.419824230788958
- Mean AUC: 0.8763721175369092

Per-label AUC:

| Label | AUC |
| --- | ---: |
| Atelectasis | 0.8424146981627297 |
| Cardiomegaly | 0.7972370766488412 |
| Consolidation | 0.8920955882352942 |
| Edema | 0.9333333333333333 |
| Pleural Effusion | 0.9167798913043478 |

## Thresholds

Per-label thresholds are selected on `archive/valid.csv` by maximizing F1.

Generated artifacts:

- `outputs/evaluation/threshold_report.csv`
- `outputs/evaluation/thresholds.json`
- `outputs/evaluation/sample_predictions.csv`

The web demo reads `outputs/evaluation/thresholds.json` when available. If thresholds are missing, the UI must show probabilities only and avoid confident classification language.

## Intended Use

This project is a research/demo web application for inspecting CheXpert-style multi-label model behavior on chest X-ray images.

## Limitations

- Not a medical device.
- Not calibrated for clinical decision-making.
- Validation set is small (`202` frontal rows).
- Thresholds are tuned on the validation split and may not generalize.
- The model should not be used to diagnose, treat, triage, or rule out disease.

## Required Disclaimer

Research prototype only. Do not use these results for medical decisions.
