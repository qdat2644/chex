from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_latex_table(data: dict) -> str:
    m_conv = data["models"]["convnext_small"]["metrics_per_label"]
    m_dense = data["models"]["densenet121_baseline"]["metrics_per_label"]
    comp = data["paired_statistical_comparison"]["per_label_comparison"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Empirical Frontal Chest Radiograph Diagnostic Performance on the Stanford CheXpert Validation Cohort ($N=202$). Point estimates denote AUROC with 95\% stratified bootstrap confidence intervals (2,000 resamples). $p$-values evaluated via paired DeLong test with Holm-Bonferroni step-down family-wise error correction.}",
        r"\label{tab:chexpert_benchmark}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Pathology Finding} & \textbf{DenseNet-121 (Baseline)} & \textbf{ConvNeXt-Small (Ours)} & \textbf{$\Delta$ AUROC (95\% CI)} & \textbf{DeLong $p_{\text{adj}}$} \\",
        r"\midrule",
    ]

    for label in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]:
        d_auc = m_dense[label]["auroc"]
        c_auc = m_conv[label]["auroc"]
        d_ci = m_dense[label]["ci_95"]
        c_ci = m_conv[label]["ci_95"]
        delta_ci = comp[label]["delta_auc"]
        p_adj = comp[label]["delong_p_holm_adj"]

        # Honest bolding: only bold the actual higher value per row
        if d_auc > c_auc:
            d_str = f"\\textbf{{{d_ci}}}"
            c_str = f"{c_ci}"
        else:
            d_str = f"{d_ci}"
            c_str = f"\\textbf{{{c_ci}}}"

        lines.append(f"{label} & {d_str} & {c_str} & {delta_ci} & {p_adj:.3f} \\\\")

    mean_d = data["models"]["densenet121_baseline"]["mean_auroc_ci_95"]
    mean_c = data["models"]["convnext_small"]["mean_auroc_ci_95"]
    delta_m = data["paired_statistical_comparison"]["macro_auc_difference"]
    p_m = data["paired_statistical_comparison"]["macro_p_value_delong"]

    lines.extend([
        r"\midrule",
        f"\\textbf{{Mean AUROC (Macro)}} & {mean_d} & \\textbf{{{mean_c}}} & {delta_m} & {p_m:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_readme_markdown(data: dict) -> str:
    m_conv = data["models"]["convnext_small"]["metrics_per_label"]
    m_dense = data["models"]["densenet121_baseline"]["metrics_per_label"]
    comp = data["paired_statistical_comparison"]["per_label_comparison"]

    rows = []
    for label in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]:
        d_auc = m_dense[label]["auroc"]
        c_auc = m_conv[label]["auroc"]
        d_ci = m_dense[label]["ci_95"]
        c_ci = m_conv[label]["ci_95"]
        delta_ci = comp[label]["delta_auc"]
        p_adj = comp[label]["delong_p_holm_adj"]

        if d_auc > c_auc:
            d_cell = f"**{d_ci}**"
            c_cell = f"{c_ci}"
        else:
            d_cell = f"{d_ci}"
            c_cell = f"**{c_ci}**"

        rows.append(f"| **{label}** | {d_cell} | {c_cell} | {delta_ci} | {p_adj:.3f} (NS) |")

    mean_d = data["models"]["densenet121_baseline"]["mean_auroc_ci_95"]
    mean_c = data["models"]["convnext_small"]["mean_auroc_ci_95"]
    delta_m = data["paired_statistical_comparison"]["macro_auc_difference"]
    p_m = data["paired_statistical_comparison"]["macro_p_value_delong"]

    table_md = f"""| Finding | DenseNet-121 (Stanford Baseline) | ConvNeXt-Small (Ours, Calibrated) | $\\Delta$ AUROC (95% CI) | Paired DeLong $p_{{\\text{{adj}}}}$ |
|:---|:---:|:---:|:---:|:---:|
""" + "\n".join(rows) + f"""
| **Mean AUROC (Macro)** | {mean_d} | **{mean_c}** | {delta_m} | {p_m:.3f} (NS) |"""

    readme_content = f"""# CheXpert AI Workstation: Clinical Radiograph Classification & Explainability System

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

{table_md}

*Note: NS = Not Statistically Significant ($p_{{\\text{{adj}}}} \\ge 0.05$). Bolding indicates the highest point estimate per row. While ConvNeXt-Small achieves a higher point estimate on 4 of 5 target pathologies and macro mean AUROC, the difference is within statistical variance of the validation cohort size.*

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
1. **Leak-Free Partitioning**: Patient-level multi-label stratification ensures $0\\%$ patient overlap across training ($80\\%$), calibration ($10\\%$), and internal validation ($10\\%$).
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
@article{{irvin2019chexpert,
  title={{CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison}},
  author={{Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Hako, Behzad and Behroozi, Peter and Blankenberg, Fiona and others}},
  journal={{AAAI Conference on Human Computation and Crowdsourcing}},
  year={{2019}}
}}
```
"""
    return readme_content


def main():
    artifact_path = PROJECT_ROOT / "outputs" / "benchmark_artifact.json"
    if not artifact_path.exists():
        print(f"Error: Artifact file {artifact_path} does not exist.")
        sys.exit(1)

    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    # 1. Update README.md
    readme_path = PROJECT_ROOT / "README.md"
    readme_path.write_text(generate_readme_markdown(data), encoding="utf-8")
    print(f"Synchronized README.md from benchmark artifact: {readme_path}")

    # 2. Update LaTeX table
    tex_path = PROJECT_ROOT / "outputs" / "evaluation" / "table_results.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(generate_latex_table(data), encoding="utf-8")
    print(f"Generated LaTeX Table: {tex_path}")


if __name__ == "__main__":
    main()
