# CheXpert AI Workstation: Clinical Radiograph Classification & Explainability System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-qdat264%2Fchexpert--convnext--small-FFD21E.svg?style=flat&logo=huggingface)](https://huggingface.co/qdat264/chexpert-convnext-small)
[![Docker](https://img.shields.io/badge/Docker-Non--Root%20Hardened-2496ED.svg?style=flat&logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Executive Summary & Clinical Intent

CheXpert AI Workstation is an open-source, reproducible deep learning system designed for multi-label chest radiograph (CXR) interpretation. Built on **ConvNeXt-Small** and trained with **Asymmetric Loss (ASL)** on the Stanford CheXpert dataset, it delivers calibrated probability estimates, high-resolution Grad-CAM visual attention overlays, and interactive PACS-grade windowing controls (WW/WC) directly in a web workstation.

> [!WARNING]
> **Investigational Use Only**: This software is an experimental research prototype. It is NOT FDA/CE-cleared for diagnostic use. All algorithmic outputs require independent clinical correlation by a board-certified radiologist.

---

## 2. Scientific Benchmark & Comparative Performance

### 2.1 Evaluation Cohort & Locked Test Protocol
All point estimates are evaluated on the official **Stanford CheXpert Frontal Radiograph Validation Cohort** ($N = 202$ studies) under a **strict leak-free locked evaluation protocol**. Optimal decision thresholds were frozen strictly on the independent internal calibration split before locked test evaluation.

Confidence intervals are generated via **stratified bootstrap resampling** (2,000 resamples, 95% percentile intervals). Model comparisons utilize the **paired DeLong test** with **Holm-Bonferroni family-wise error rate correction**.

| Finding | DenseNet-121 (Stanford Baseline) | ConvNeXt-Small (Ours, Calibrated) | $\Delta$ AUROC (95% CI) | Paired DeLong $p_{\text{adj}}$ |
|:---|:---:|:---:|:---:|:---:|
| **Atelectasis** | **0.8424 (0.7810–0.8990)** | 0.8142 (0.7480–0.8750) | -0.0282 (-0.0650 to +0.0080) | 0.500 (NS) |
| **Cardiomegaly** | 0.8885 (0.8310–0.9390) | **0.9026 (0.8520–0.9480)** | +0.0141 (-0.0150 to +0.0430) | 0.680 (NS) |
| **Consolidation** | 0.9188 (0.8680–0.9620) | **0.9442 (0.9050–0.9780)** | +0.0254 (-0.0050 to +0.0560) | 0.490 (NS) |
| **Edema** | 0.9126 (0.8600–0.9570) | **0.9168 (0.8670–0.9610)** | +0.0042 (-0.0220 to +0.0310) | 0.740 (NS) |
| **Pleural Effusion** | 0.8738 (0.8190–0.9250) | **0.8940 (0.8410–0.9430)** | +0.0202 (-0.0110 to +0.0510) | 0.630 (NS) |
| **Mean AUROC (Macro)** | 0.8872 (0.8520–0.9180) | **0.8944 (0.8610–0.9250)** | +0.0072 (-0.0080 to +0.0224) | 0.354 (NS) |

*Note: NS = Not Statistically Significant ($p_{\text{adj}} \ge 0.05$). Bolding indicates the highest point estimate per row. While ConvNeXt-Small achieves a higher point estimate on 4 of 5 target pathologies and macro mean AUROC, the difference is within statistical variance of the validation cohort size.*

---

## 3. Core Architecture & Pipeline

```
Raw DICOM / Image
  │
  ├─► Medical Image Preprocessing Pipeline (Modality LUT -> VOI LUT -> Inversion -> Normalization)
  │
  ├─► ConvNeXt-Small Backbone (7x7 Depthwise Conv, LayerNorm, GELU)
  │     │
  │     ├─► Calibrated Multi-Label Classification Head (5 Competition Pathologies)
  │     └─► Target Layer Grad-CAM (Noise-Floor Filtered & Adaptive Alpha Scaling)
  │
  └─► PACS DICOM Viewer + Interactive Grad-CAM Selector + Instant Bilingual Report
```

### 3.1 Methodological Innovations
1. **Leak-Free Partitioning**: Patient-level multi-label stratification ensures $0\%$ patient overlap across training ($80\%$), calibration ($10\%$), and internal validation ($10\%$).
2. **Masked Uncertainty Loss**: Label `-1` (uncertain) under the `ignore` policy is strictly masked out from both numerator loss and denominator reduction, preventing artificial bias.
3. **Deterministic PHI Anonymization**: Protected Health Information (Patient ID, Patient Name) is masked using HMAC-SHA256 with an ephemeral high-entropy secret.
4. **Single-Source-of-Truth Governance**: All metrics, SHA-256 hashes, and configuration parameters are synchronized from `outputs/benchmark_artifact.json`.

---

## 4. Production Security & Deployment

- **Non-Root Hardened Container**: Runs under unprivileged user `chexpert` (UID 10001) with `no-new-privileges:true`.
- **Fail-Closed Artifact Integrity**: Refuses startup if SHA-256 checksums of the model checkpoint or thresholds JSON do not match pinned values.
- **Docker Secrets Integration**: Zero hardcoded secrets in repository or image layers.
- **Rate Limiting & Audit Logging**: 60 requests/min rate limiter and structured audit logging with unique `X-Request-ID`.

```bash
# Clone and launch in one command
git clone https://github.com/qdat2644/chex.git
cd chex
docker compose up --build
```

---

## 5. Artifact & Checksum Ledger

| Artifact Name | Relative Path | SHA-256 Checksum |
|:---|:---|:---|
| **ConvNeXt-Small Checkpoint** | `checkpoints/chexpert_convnext_small.pt` | `b8b1884a911f6ff9408de141c48034a39f4515e2c8deaadd25312e07601c9bc0` |
| **Calibrated Thresholds** | `outputs/evaluation/thresholds.json` | `7ad370d14f7941fae476d0f2ba2038fcfeedbbf70dcfd1e45cb083baa28965ea` |
| **Benchmark Manifest** | `outputs/benchmark_artifact.json` | *(Version 2.0.0 Single-Source)* |

---

## 6. Citation & References

```bibtex
@article{irvin2019chexpert,
  title={CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Hako, Behzad and Behroozi, Peter and Blankenberg, Fiona and others},
  journal={AAAI Conference on Human Computation and Crowdsourcing},
  year={2019}
}
```
