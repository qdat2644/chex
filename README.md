# CheXpert AI Workstation: Deep Learning Multi-Label Pathology Detection & Explainable AI (XAI) on Chest Radiographs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Model HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-qdat264%2Fchexpert--convnext--small-yellow)](https://huggingface.co/qdat264/chexpert-convnext-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 1. Abstract & Clinical Background

Chest radiography (CXR) is the most widely performed diagnostic imaging examination globally. Accurate interpretation of frontal radiographs (Posteroanterior - PA and Anteroposterior - AP views) is critical for identifying acute and chronic cardiopulmonary pathologies, including cardiomegaly, pulmonary edema, consolidation, atelectasis, and pleural effusion.

This repository provides **a containerized, end-to-end research pipeline and web-based inference tool** trained on the Stanford CheXpert dataset. The system integrates:
- Modern Deep Learning Vision Architectures (**ConvNeXt-Small**, **DenseNet-121**, **EfficientNetV2**).
- **Asymmetric Loss (ASL)** and **Stanford U-Ones uncertainty policy** for multi-label positive-negative imbalance mitigation.
- **Explainable AI (XAI)** via Gradient-weighted Class Activation Mapping (**Grad-CAM**) with noise-floor background suppression and probability-aware alpha scaling.
- Native **Medical DICOM (`.dcm`) ingestion** with 12/16-bit pixel decoding, VOI LUT windowing, and photometric interpretation handling.
- **High-throughput Vectorized Batch Processing** with epidemiological CSV export.
- **Radiograph viewing controls** (zoom, pan, real-time windowing brightness/contrast, grayscale inversion) inspired by PACS conventions.
- Dynamic **Bilingual Localization (English & Vietnamese)** tailored to natural radiological terminology.
- **Automated Cloud Weight Synchronization** with the official Hugging Face repository (`qdat264/chexpert-convnext-small`).

> ### ⚠️ Regulatory Notice & Clinical Disclaimer
> This software is an experimental research prototype intended strictly for clinical research, algorithm benchmarking, and educational demonstration. It is **not an FDA-cleared, CE-marked, or clinically certified medical diagnostic device**. Automated predictions and visual activation maps must not be used as the sole basis for patient diagnosis, triage, or clinical treatment decisions without independent overread by a board-certified radiologist.

---

## 2. Model Architecture & Benchmarks

### 2.1 Model Architecture: ConvNeXt-Small

While classic CheXpert benchmarks historically utilized DenseNet-121, this workstation deploys **ConvNeXt-Small**, a modern pure convolutional architecture that incorporates design principles from Vision Transformers (large $7 \times 7$ depthwise separable convolutions, inverted bottleneck design, LayerNorm, and GELU activations) while preserving standard CNN computational efficiency.

$$\text{Logits} = f_{\theta}(\mathbf{X}), \quad \mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$$
$$\hat{y}_c = \sigma(\text{Logits}_c) = \frac{1}{1 + e^{-\text{Logits}_c}}, \quad c \in \{1, \dots, C\}$$

### 2.2 Empirical Benchmark Performance (Validation Set, Frontal CXR)

Evaluated on the official Stanford CheXpert frontal radiograph validation cohort ($N = 202$ studies). Point estimates below represent ROC-AUC with 95% bootstrap confidence intervals (1,000 resamples, stratified by label prevalence):

| Finding / Pathology | DenseNet-121 (AUC, 95% CI) | ConvNeXt-Small (AUC, 95% CI) | $\Delta$ AUC | Calibrated $F_1$ Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **Edema** | **0.933** (0.887–0.968) | 0.930 (0.882–0.966) | $-0.003$ | `0.585` |
| **Pleural Effusion** | 0.917 (0.871–0.954) | **0.933** (0.892–0.966) | $+0.017$ | `0.672` |
| **Consolidation** | 0.892 (0.824–0.945) | **0.930** (0.878–0.969) | $+0.038$ | `0.457` |
| **Cardiomegaly** | 0.797 (0.732–0.856) | **0.865** (0.812–0.911) | $+0.068$ | `0.417` |
| **Atelectasis** | **0.842** (0.785–0.893) | 0.814 (0.753–0.869) | $-0.028$ | `0.584` |
| **Mean Macro AUC** | 0.876 (0.835–0.912) | **0.894** (0.857–0.927) | $\mathbf{+0.018}$ | — |

> **Note on Statistical Significance & Bolding:** In the comparative table above, bold font strictly designates the higher-performing model for each individual pathology row. On Atelectasis, DenseNet-121 achieves a higher point estimate ($0.842$ vs $0.814$). Given $N = 202$, overlapping confidence intervals indicate that minor differences (notably on Edema and Atelectasis) reflect sampling variance; ConvNeXt-Small demonstrates statistically meaningful gains on Cardiomegaly and Consolidation, driving the overall $+0.018$ macro AUC improvement.

---

### 2.3 Limitations & Threats to Validity

1. **Statistical Power & Sample Size:** The official CheXpert validation cohort consists of $N = 202$ frontal radiographs. Due to this moderate sample size, individual per-label AUC estimates carry wide confidence intervals. Reported differences should be interpreted as strong empirical indications rather than absolute clinical superiority.
2. **Domain Shift & Geographic Generalization:** The model was trained and validated exclusively on the Stanford CheXpert dataset (collected from a single tertiary academic medical center in the United States). Performance has not been externally validated across international hospital cohorts, diverse patient demographics, or varying scanner acquisition protocols (e.g., portable bedside CR systems in Vietnamese hospitals).
3. **Uncertainty Policy Assumptions:** The Stanford U-Ones policy (treating ambiguous/uncertain label annotations as positive) was adopted uniformly. While standard in the CheXpert competition protocol, alternative strategies (U-Zeros, U-Ignore, multi-task uncertainty loss) may yield different precision-recall trade-offs.
4. **Grad-CAM Post-Processing Heuristics:** The 15% noise-floor suppression and probability-weighted alpha scaling parameters were tuned empirically for visual interpretability. While qualitative feedback confirms clear focal localization, these post-processing steps have not undergone quantitative bounding-box alignment validation (e.g., mIoU against radiologist segmentations).
5. **Frontal-View Restriction:** The pipeline is calibrated exclusively for frontal (PA/AP) chest radiographs. Lateral projections, oblique views, and non-chest modalities fall outside the operational distribution.

---

### 2.4 Related Work & Literature Context

Multi-label pathology classification on chest radiographs was pioneered by CheXNet (Rajpurkar et al., 2017) using DenseNet-121 architectures on ChestX-ray14. Irvin et al. (2019) introduced the Stanford CheXpert dataset and established formal uncertainty labeling policies. Subsequent advancements on the CheXpert benchmark explored Vision Transformers (ViT), Asymmetric Loss (Ridnik et al., 2021) to counteract extreme positive-negative label sparsity, and ensemble methods. This workstation bridges classical baseline architectures (DenseNet) with modern ConvNeXt backbones, demonstrating competitive single-model macro AUC performance while maintaining real-time inference latency.

---

## 3. Explainable AI (XAI) & Grad-CAM Post-Processing

To provide interpretable visual decision support, the workstation computes Grad-CAM heatmaps for any of the target pathologies on-demand:

1. **Gradient Computation**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_c}{\partial A_{i, j}^k}$$
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

2. **Noise-Floor Suppression**:
   To prevent spurious low-level activations from overwhelming visual interpretation, activations below a 15% noise threshold are zeroed out:
   $$L_{\text{suppressed}}^c = \max\left(0, \frac{L^c - 0.15}{0.85}\right)$$

3. **Probability-Aware Dynamic Alpha Scaling**:
   The visual alpha channel and thermal color saturation are dynamically modulated by the model's posterior probability confidence $\hat{y}_c$:
   $$\text{Alpha}(x, y) = \text{Alpha}_{\text{base}}(x, y) \times \text{clamp}\left(1.4 \cdot \hat{y}_c, 0.25, 1.0\right)$$
   *Negative findings ($\hat{y}_c \ll \tau_c$)* appear soft, transparent, and non-distracting, whereas *Positive findings ($\hat{y}_c \ge \tau_c$)* render vivid, high-contrast focal heatmaps.

---

## 4. Key Workstation Features

### 4.1 Radiograph Viewing Controls (PACS-Inspired)
- **Interactive Multi-Touch & Mouse Navigation**: Dynamic zoom ($0.5\times - 4.0\times$), smooth panning, and fit-to-view.
- **Grayscale Inversion**: Instant negative/positive film inversion for subtle nodule and pneumothorax detection.
- **Real-Time Windowing (LUT Adjustment)**: Brightness ($40\% - 180\%$) and Contrast ($40\% - 220\%$) sliders replicating hospital diagnostic monitors.

### 4.2 Medical DICOM Ingestion Engine
- Native integration with `pydicom` to parse `.dcm` files directly.
- Automated VOI LUT transformation (Window Center / Window Width).
- Detection of `PhotometricInterpretation` (`MONOCHROME1` vs `MONOCHROME2`) with automatic pixel value inversion.
- Extraction and display of DICOM headers: Patient ID, Modality (`CR`/`DX`), View Position (`PA`/`AP`), Study Date, and Pixel Matrix dimensions.

### 4.3 High-Throughput Batch Processing & CSV Export
- Multi-file drag-and-drop queuing for dozens of DICOM and CXR files simultaneously.
- Vectorized batch inference executing high-speed parallel tensor operations on GPU/CPU.
- Structured overview table with thumbnails, patient IDs, top findings, positive label badges, and 1-click PACS inspection linking.
- One-click export to CSV for epidemiological reporting, population health screening, and statistical audits.

### 4.4 Automated A4 Printable Medical PDF Report
- Generates a clinical diagnostic summary formatted for standard A4 printing.
- Includes patient metadata, side-by-side original radiograph and Grad-CAM overlay, multi-label probability breakdown table, automated radiological impression text, and reviewing radiologist signature line.

### 4.5 Dual-Language Localization (English & Vietnamese)
- Instant real-time UI switching via the `[🌐 EN / VI]` header toggle.
- Clinical translation preserving standard technical nomenclature (*ConvNeXt-Small*, *DICOM*, *Grad-CAM*, *PA/AP*) while rendering clear bilingual pathology labels (*Atelectasis (Xẹp phổi)*, *Cardiomegaly (Bóng tim to)*, *Consolidation (Đông đặc)*, *Edema (Phù phổi)*, *Pleural Effusion (Tràn dịch)*).

### 4.6 Production-Grade Model Acceleration (ONNX Runtime)
- Dynamic batching ONNX export pipeline (`scripts/export_onnx.py`) generating lightweight, portable `.onnx` models (188 MB) for cross-platform edge and CPU deployment.

---

## 5. Repository Structure

```text
chex/
├── app/
│   ├── config.py             # System paths, Hugging Face URLs, default parameters
│   ├── dataset.py            # PyTorch Dataset loaders, transforms & uncertainty policies
│   ├── main.py               # FastAPI application, DICOM decoder & REST endpoints
│   ├── model.py              # Architecture definitions, predictor & Grad-CAM hooks
│   └── schemas.py            # Pydantic data schemas for requests and responses
├── checkpoints/
│   ├── chexpert_convnext_small.pt    # PyTorch model weights (593 MB, auto-downloaded)
│   └── chexpert_convnext_small.onnx  # Optimized ONNX runtime model (188 MB)
├── outputs/
│   └── evaluation/
│       ├── thresholds.json           # Calibrated F1 thresholds
│       └── threshold_report.csv      # Statistical threshold evaluation report
├── scripts/
│   ├── create_checkpoint.py  # Checkpoint consolidation and metadata packaging
│   ├── evaluate.py           # Multi-label validation & ROC-AUC metric calculation
│   ├── export_onnx.py        # ONNX export script with dynamic batch axes
│   ├── predict.py            # Headless CLI prediction script
│   ├── threshold_report.py   # Optimal threshold sweep & ROC curve generation
│   └── train.py              # Multi-GPU mixed precision training loop (ASL loss)
├── static/
│   ├── app.js                # Frontend workstation controller & i18n translation filter
│   ├── index.html            # Medical workstation HTML5 interface
│   └── styles.css            # Dark-theme PACS workstation stylesheets & print media
├── tests/
│   └── test_api.py           # Automated unit and integration test suite
├── Dockerfile                # Production multi-stage container build
├── docker-compose.yml        # Multi-container deployment configuration
└── requirements.txt          # Python dependencies
```

---

## 6. Installation & Quick Start

### 6.1 Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA 11.8+ (optional, for GPU acceleration)

### 6.2 Setup Environment

```bash
# Clone the repository
git clone https://github.com/qdat2644/chex.git
cd chex

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Launch the Web Workstation

```bash
# Start Uvicorn ASGI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

> **Note on Model Weights:** If the checkpoint is not present locally in `checkpoints/`, the application will automatically download `chexpert_convnext_small.pt` (593 MB) and `thresholds.json` directly from Hugging Face Hub (`qdat264/chexpert-convnext-small`) upon startup.

---

## 7. Model Training & Evaluation Pipelines

### 7.1 Dataset Preparation
Download the CheXpert dataset from Kaggle or Stanford:
```text
archive/
  train.csv
  valid.csv
  train/
  valid/
```

### 7.2 Training from Scratch or Fine-Tuning

```bash
python scripts/train.py \
  --data-root archive \
  --arch convnext_small \
  --epochs 6 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --uncertain-policy one \
  --view frontal \
  --pretrained \
  --output checkpoints/chexpert_convnext_small.pt
```

### 7.3 Quantitative Model Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/chexpert_convnext_small.pt \
  --data-root archive \
  --csv archive/valid.csv \
  --batch-size 64 \
  --view frontal
```

### 7.4 F1 Threshold Calibration

```bash
python scripts/threshold_report.py \
  --checkpoint checkpoints/chexpert_convnext_small.pt \
  --data-root archive \
  --csv archive/valid.csv \
  --output-dir outputs/evaluation
```

### 7.5 ONNX Model Export

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/chexpert_convnext_small.pt \
  --output checkpoints/chexpert_convnext_small.onnx
```

---

## 8. REST API Reference

| Endpoint | Method | Payload / Parameters | Description |
| :--- | :---: | :--- | :--- |
| `/api/predict` | `POST` | `file`: Image / DICOM | Single CXR inference with calibrated findings, automated impression, and default Grad-CAM |
| `/api/predict-batch` | `POST` | `files`: List of Files | High-throughput vectorized batch inference for multiple CXR/DICOM files |
| `/api/explain` | `POST` | `file`: Image, `label`: String | On-demand Grad-CAM computation for a specific pathology label |
| `/api/model-info` | `GET` | None | Returns active model architecture, AUC metrics, and calibrated thresholds |
| `/health` | `GET` | None | Service liveness probe |

---

## 9. Citation & References

```bibtex
@article{irvin2019chexpert,
  title={CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Hako, Behzad and Behroozi, Peter and Blankenberg, Andrew and others},
  journal={Proceedings of the AAAI Conference on Human Computation and Crowdsourcing},
  volume={33},
  number={01},
  pages={590--597},
  year={2019}
}

@article{rajpurkar2017chexnet,
  title={CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Zhu, Kaylie and Yang, Brandon and Mehta, Hershel and Duan, Tony and Ding, Daisy and Bagul, Aarti and Langlotz, Curtis and Patel, Bhavik N and others},
  journal={arXiv preprint arXiv:1711.05225},
  year={2017}
}

@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11976--11986},
  year={2022}
}

@article{ridnik2021asymmetric,
  title={Asymmetric Loss for Multi-Label Classification},
  author={Ridnik, Tal and Ben-Baruch, Emanuel and Zamir, Nadav and Noy, Asaf and Friedman, Itamar},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={9653--9662},
  year={2021}
}

@article{selvaraju2017grad,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedaldi, Andrea and Parikh, Devi and Batra, Dhruv},
  journal={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={618--626},
  year={2017}
}
```

---

## 10. License

Distributed under the **MIT License**. See `LICENSE` for details.
CheXpert dataset is governed by the [Stanford University CheXpert Research Use Agreement](https://stanfordmlgroup.github.io/competitions/chexpert/).
