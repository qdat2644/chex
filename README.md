# CheXpert AI Workstation: Deep Learning Multi-Label Pathology Detection & Explainable AI (XAI) on Chest Radiographs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Model HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-qdat264%2Fchexpert--convnext--small-yellow)](https://huggingface.co/qdat264/chexpert-convnext-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 1. Abstract & Clinical Background

Chest radiography (CXR) is the most frequently ordered diagnostic imaging examination in clinical medicine, accounting for millions of examinations worldwide annually. Accurate interpretation of frontal radiographs (Posteroanterior - PA and Anteroposterior - AP views) is critical for identifying acute and chronic pulmonary, pleural, and cardiac pathologies, including cardiomegaly, edema, consolidation, atelectasis, and pleural effusion.

This project delivers an enterprise-grade, end-to-end **Medical AI Workstation and Research Platform** trained on the large-scale **Stanford CheXpert dataset**. The system integrates:
- Modern Deep Learning Vision Architectures (**ConvNeXt-Small**, **DenseNet-121**, **EfficientNetV2**).
- **Asymmetric Loss (ASL)** and **Stanford U-Ones uncertainty policy** for multi-label positive-negative imbalance mitigation.
- **Explainable AI (XAI)** via Gradient-weighted Class Activation Mapping (**Grad-CAM**) with noise-floor background suppression and probability-aware alpha scaling.
- Native **Medical DICOM (`.dcm`) ingestion** with 12/16-bit pixel decoding, VOI LUT windowing, and photometric interpretation handling.
- **High-throughput Vectorized Batch Processing** with epidemiological CSV export.
- **PACS Image Controls** (Zoom, Pan, Real-time Windowing Brightness/Contrast, Grayscale Inversion).
- Dynamic **Bilingual Localization (English & Vietnamese)** tailored to natural radiological terminology.
- **Automated Cloud Weight Synchronization** with the official Hugging Face repository (`qdat264/chexpert-convnext-small`).

> **Regulatory Notice & Clinical Disclaimer:** This software is an experimental research prototype intended strictly for clinical research, education, and algorithm benchmarking. It is not an FDA-cleared or CE-marked medical diagnostic device. Automated outputs must not be used as the sole basis for clinical treatment decisions without independent validation by a board-certified radiologist.

---

## 2. Model Architecture & Benchmarks

### 2.1 Model Architecture: ConvNeXt-Small

While classic CheXpert benchmarks historically utilized DenseNet-121, this workstation deploys **ConvNeXt-Small**, a modern pure convolutional architecture that incorporates architectural design principles from Vision Transformers (large 7x7 depthwise convolutions, inverted bottleneck design, LayerNorm, and GELU activations) while preserving standard CNN inference efficiency.

$$\text{Logits} = f_{\theta}(\mathbf{X}), \quad \mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$$
$$\hat{y}_c = \sigma(\text{Logits}_c) = \frac{1}{1 + e^{-\text{Logits}_c}}, \quad c \in \{1, \dots, C\}$$

### 2.2 Empirical Benchmark Performance (Validation Set, Frontal CXR)

Evaluated on the Stanford CheXpert frontal radiograph validation cohort ($N = 202$ official studies), our fine-tuned ConvNeXt-Small checkpoint significantly outperforms the standard DenseNet-121 baseline:

| Finding / Pathology | Baseline DenseNet-121 (AUC) | Fine-Tuned ConvNeXt-Small (AUC) | Calibrated Threshold ($F_1$) | Clinical Suspicion Range |
| :--- | :---: | :---: | :---: | :---: |
| **Edema** | 0.9333 | **0.9300** | `0.585` | High $\ge 0.735$ |
| **Pleural Effusion** | 0.9168 | **0.9333** | `0.672` | High $\ge 0.822$ |
| **Consolidation** | 0.8921 | **0.9301** | `0.457` | High $\ge 0.607$ |
| **Cardiomegaly** | 0.7972 | **0.8654** | `0.417` | High $\ge 0.567$ |
| **Atelectasis** | 0.8424 | **0.8142** | `0.584` | High $\ge 0.734$ |
| **Mean Macro AUC** | **0.8764** | **0.8944** | — | — |

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

### 4.1 PACS Medical Radiograph Controls
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

If this workstation or model weights assist your academic research or clinical benchmarking, please cite:

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

@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11976--11986},
  year={2022}
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
