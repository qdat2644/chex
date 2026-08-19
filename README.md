# CheXpert AI Workstation: Clinical Radiograph Classification & Explainability System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-qdat264%2Fchexpert--convnext--small-FFD21E.svg?style=flat&logo=huggingface)](https://huggingface.co/qdat264/chexpert-convnext-small)
[![Docker](https://img.shields.io/badge/Docker-Non--Root%20Hardened-2496ED.svg?style=flat&logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Executive Summary & Clinical Intent

CheXpert AI Workstation is an open-source, reproducible deep learning system designed for multi-label chest radiograph (CXR) interpretation. Built on **ConvNeXt-Small** and trained with **Asymmetric Loss (ASL)**, it delivers probability estimates, Grad-CAM visual attention overlays, and PACS-grade windowing controls (WW/WC) directly in a web workstation.

> [!WARNING]
> **Investigational Use Only**: This software is an experimental research prototype. It is NOT FDA/CE-cleared for diagnostic use. All algorithmic outputs require independent clinical correlation by a board-certified radiologist.

---

## 2. Scientific Benchmark & Comparative Performance (Protocol v0.1)

### 2.1 Locked Evaluation Status
All historical unverified metrics have been moved to `outputs/legacy/` in accordance with the leak-free scientific protocol specification. Formal locked test evaluation will be performed following experiment freeze across 5 fixed seeds (`seeds 42..46`).

| Finding | DenseNet-121 (Baseline) | ConvNeXt-Small (Ours) | $\Delta$ AUROC (95% CI) | Paired DeLong $p_{\text{adj}}$ |
|:---|:---:|:---:|:---:|:---:|
| **Atelectasis** | TBD | TBD | TBD | TBD |
| **Cardiomegaly** | TBD | TBD | TBD | TBD |
| **Consolidation** | TBD | TBD | TBD | TBD |
| **Edema** | TBD | TBD | TBD | TBD |
| **Pleural Effusion** | TBD | TBD | TBD | TBD |
| **Mean AUROC (Macro)** | TBD | TBD | TBD | TBD |

*Status: No protocol-compliant final results available yet. Benchmark tables will be populated automatically upon execution of `scripts/sync_reports.py` with a valid `outputs/final/protocol_v0_1/benchmark_artifact.json`.*

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

### 3.1 Methodological Rigor
1. **Patient-Level Stratification**: Iterative multi-label stratification ensures $0\%$ patient overlap across training ($80\%$), calibration ($10\%$), and internal validation ($10\%$).
2. **Masked Uncertainty Loss**: Label `-1` (uncertain) under the `ignore` policy is strictly masked out from both numerator loss and denominator reduction, preventing artificial bias.
3. **Deterministic PHI Anonymization**: Protected Health Information (Patient ID, Patient Name) is masked using HMAC-SHA256 with an ephemeral high-entropy secret.
4. **Single-Source-of-Truth Governance**: All metrics, SHA-256 hashes, and configuration parameters are synchronized from `outputs/final/protocol_v0_1/benchmark_artifact.json`.

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
| **Protocol Config** | `configs/protocol_v0_1.yaml` | *(Protocol Version 0.1)* |

---

## 6. Citation & References

```bibtex
@article{irvin2019chexpert,
  title={{CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison}},
  author={{Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Hako, Behzad and Behroozi, Peter and Blankenberg, Fiona and others}},
  journal={{AAAI Conference on Human Computation and Crowdsourcing}},
  year={{2019}}
}
```
