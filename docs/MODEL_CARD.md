# CheXpert AI Model Card: ConvNeXt-Small

## 1. Model Details

- **Architecture:** ConvNeXt-Small (Liu et al., CVPR 2022)
- **Primary Checkpoint:** `checkpoints/chexpert_convnext_small.pt` (PyTorch) / `checkpoints/chexpert_convnext_small.onnx` (ONNX Runtime)
- **Hugging Face Hub:** [`qdat264/chexpert-convnext-small`](https://huggingface.co/qdat264/chexpert-convnext-small)
- **Task:** Multi-Label Pathology Classification on Frontal Chest Radiographs (CXR)
- **Loss Formulation:** Asymmetric Loss (ASL) for positive-negative class imbalance
- **Input Preprocessing:** Resize to $224 \times 224$, 3-channel broadcast, standard ImageNet normalization $(\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225])$

---

## 2. Target Pathologies & Label Order

The model outputs multi-label posterior probabilities across the 5 canonical CheXpert competition findings:

1. **Atelectasis**
2. **Cardiomegaly**
3. **Consolidation**
4. **Edema**
5. **Pleural Effusion**

---

## 3. Empirical Validation & Metrics

Evaluated on the official Stanford CheXpert frontal radiograph validation cohort ($N = 202$ studies):

- **Mean Macro ROC-AUC:** **`0.8944`**

| Finding / Pathology | ROC-AUC | Optimal Calibrated Threshold ($F_1$) |
| :--- | :---: | :---: |
| **Edema** | `0.9300` | `0.585` |
| **Pleural Effusion** | `0.9333` | `0.672` |
| **Consolidation** | `0.9301` | `0.457` |
| **Cardiomegaly** | `0.8654` | `0.417` |
| **Atelectasis** | `0.8142` | `0.584` |

---

## 4. Intended Use & Limitations

- **Intended Use:** Clinical research benchmarking, radiology algorithm interpretation, educational demonstration of Explainable AI (Grad-CAM) in chest imaging.
- **Limitations:**
  - Designed strictly for **frontal (PA/AP)** radiographs; lateral views or non-CXR modalities are out-of-distribution.
  - Not an FDA-cleared medical device. Overread by a licensed radiologist is mandatory.
