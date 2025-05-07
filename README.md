# Reproducing "Hidden Stratification Causes Clinically Meaningful Failures in Medical Imaging" (NeurIPS ML4H 2019)

## 📌 Overview

This repository reproduces key findings from the paper:
**“Hidden Stratification Causes Clinically Meaningful Failures in Medical Imaging”**

NeurIPS ML4H 2019 · [arXiv link](https://arxiv.org/abs/1909.12475)
@article{oakden2019hidden,
  title={Hidden stratification causes clinically meaningful failures in machine learning for medical imaging},
  author={Oakden-Rayner, Luke and Beam, Andrew L and Palmer, Lyle J},
  journal={arXiv preprint arXiv:1909.12475},
  year={2019}
}

The paper explores *hidden stratification* — clinically important but unlabeled subclasses within standard diagnostic categories — and how they can silently degrade model reliability in medical imaging. This repo attempts to replicate the findings using two datasets: **CXR14** and **MURA**.

---

## 📂 Contents

* `CRX14.ipynb`: Reproduces analysis on chest X-ray (CXR14) dataset, focusing on pneumothorax detection and subclass impact of **chest drains** and **airspace opacity**.
* `MURA_formatted.ipynb`: Implements model training and evaluation for musculoskeletal X-rays from MURA, using DenseNet-169.
* `Introduction.docx`: Full reproduction report, including methodology, evaluation, and extension analysis.

---

## 🧪 Reproducibility Summary

| Component          | Reproduced? | Notes                                                                  |
| ------------------ | ----------- | ---------------------------------------------------------------------- |
| Model Architecture | ✅           | DenseNet-121 and DenseNet-169 pretrained models implemented in PyTorch |
| Training Pipeline  | ✅           | Binary classification tasks with BCE loss, Adam optimizer              |
| AUC Results        | ✅           | Achieved AUC ≈ 0.87 on both datasets                                   |
| Subclass Analysis  | ⚠️ Partial  | Manual subclass labels (e.g., chest drains) not fully accessible       |
| Clustering Method  | ✅           | KMeans clustering applied to uncover latent failure modes              |

---

## 📊 Datasets

### 1. MURA (Musculoskeletal Radiographs)

* 40,000+ radiographic images
* Task: Normal vs Abnormal classification
* Used: Predefined train/test split
* [Dataset link](https://stanfordmlgroup.github.io/competitions/mura/)

### 2. NIH ChestX-ray14 (CXR14)

* 112,000+ frontal-view chest X-rays
* Task: Pneumothorax detection
* Used: Balanced train set; expert-labeled validation/test sets
* [Dataset link](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)

---

## ⚙️ Environment

* Python 3.9.21
* PyTorch 2.5.1, Torchvision 0.20.1
* Numpy, Pandas, Pillow, Matplotlib, Seaborn, scikit-learn, tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

You can also recreate the environment with:

```bash
conda create -n hidden-stratification python=3.9
conda activate hidden-stratification
pip install -r requirements.txt
```

---

## 🚀 Training

DenseNet-121 (CXR14):

```python
# Batch size: 128, Epochs: 2, LR: 1e-4
```

DenseNet-169 (MURA):

```python
# Batch size: 64, Epochs: 2, LR: 1e-4
```

Both use `BCEWithLogitsLoss` and Adam optimizer.

---

## 📊 Key Results

### Chest Drain Bias (CXR14):

* With drains: AUC ≈ 0.94
* Without drains: AUC ≈ 0.77
* Overall: AUC ≈ 0.87
  🔍 Manual review revealed shortcut learning by associating tubes with pneumothorax.

### Airspace Opacity:

* With opacity: AUC ≈ 0.83
* Without opacity: AUC ≈ 0.91
  📉 Opacity masks visual cues, reducing model performance.

### Unsupervised Clustering (KMeans):

* Clusters 0 & 4 had high false positives — potentially capturing confounding visual artifacts.
* Cluster 3 had 100% true negatives — indicating clear, unambiguous cases.

---

## 🔬 Limitations & Extensions

* Manual subclass annotations not available publicly.
* Clustering and saliency-based explainability can help identify hidden stratification without labels.
* Segmentation or metadata filtering may improve robustness.

---

## 📄 License

MIT License
