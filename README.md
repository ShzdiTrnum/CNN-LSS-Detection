# CNN-LSS-Detection

> Nerve Root Compression Analysis to Find Lumbar Spine Stenosis on MRI Using CNN, a custom convolutional neural network for automated four-class grading of foraminal lumbar spine stenosis (Normal / Mild / Moderate / Severe) from axial-view MRI images.

---

## Paper

**Title:** Nerve Root Compression Analysis to Find Lumbar Spine Stenosis on MRI Using CNN

**Published:** *Diagnostics* **2023**, 13, 2975. MDPI (Open Access — CC BY 4.0)
**DOI:** https://doi.org/10.3390/diagnostics13182975

---

## Table of Contents

- [Project Description](#project-description)
- [Clinical Background](#clinical-background)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Citation](#citation)
- [License](#license)

---

## Project Description

This repository implements a custom CNN model for automated classification of **foraminal lumbar spine stenosis (LSS)** from axial-view MRI images into four severity grades: Normal, Mild, Moderate, and Severe.

The pipeline extracts four Regions of Interest (ROIs) — Intervertebral Disc (IVD), Posterior Element (PE), Thecal Sac (TS), and Anteroposterior Diameter (AAP) — and trains the CNN on both Multi-ROI (all four regions) and Single-ROI (AAP only) datasets. Data augmentation (translation, zoom, rotation) is applied to address the limited size of medical imaging datasets.

---

## Clinical Background

**Lumbar Spine Stenosis (LSS)** is a narrowing of the spinal column or vertebral foramina that compresses the thecal sac and posterior nerve roots. It is a major cause of chronic low back pain (CLBP), affecting 50–80% of adults at some point in their lives.

LSS diagnosis involves analysing the anteroposterior diameter and foraminal widths in axial MRI images. Radiologists grade stenosis as:

| Grade | Description |
|-------|-------------|
| **Normal** | No stenosis |
| **Mild** | Minor narrowing, no nerve root compression |
| **Moderate** | Significant narrowing with partial compression |
| **Severe** | Complete compression of nerve roots |


## Dataset

**Sudirman et al. (2019). Lumbar Spine MRI Dataset. Mendeley Data.**
DOI: 10.17632/k57fr854j2.2

- **Patients:** 515 patients with back pain symptoms
- **Images:** 1,545 axial-view MRI images (L3–L5 levels: L3-D3, L4-D4, L5-D5)
- **Resolution:** 320 × 320 px (cropped to 240 × 240 px for training)
- **Views:** Axial (top-down) MRI slices
- **Labels:** Normal, Mild, Moderate, Severe (graded by expert radiologist)
- **Augmentation:** Three dataset sizes produced — 5k, 10k, 12.5k images — using translation, zoom (range 0.7–1.3), and rotation (±20°)

The dataset is publicly available at: https://data.mendeley.com/datasets/k57fr854j2/2

---

## Repository Structure

```
cnn-lss-detection/
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py     # Dataset loading, preprocessing, train/test split
│   ├── model.py          # Custom CNN architecture definition (Table 1)
│   ├── train.py          # Training entry point (importable + CLI)
│   ├── evaluate.py       # Metrics, plots, confusion matrix, comparison charts
│   └── inference.py      # Load model and predict on new images
│
├── configs/
│   └── training_configs.md   # Dataset size configurations and hyperparameters
│
├── data/
│   └── README.md             # Dataset download and preparation instructions
│
├── docs/
│   └── architecture_notes.md # CNN architecture details and design rationale
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| TensorFlow / Keras | ≥ 2.8 | CNN model, training, inference |
| numpy | ≥ 1.21 | Array operations |
| opencv-python | ≥ 4.5 | Image loading and preprocessing |
| scikit-learn | ≥ 1.0 | Train/test split, classification report |
| matplotlib | ≥ 3.4 | Training curves and comparison plots |
| seaborn | ≥ 0.11 | Confusion matrix heatmap |

---

## Installation

### Google Colab (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow numpy opencv-python scikit-learn matplotlib seaborn -q

import sys
sys.path.insert(0, '/content')   # so 'from src.xxx import' works
```

### Local

```bash
git clone https://github.com/ShzdiTrnum/CNN-LSS-Detection.git
cd CNN-LSS-Detection
pip install -r requirements.txt
```

## Citation

If you use this code or the findings from this paper, please cite:

```bibtex
@article{shahzadi2023lss,
  title   = {Nerve Root Compression Analysis to Find Lumbar Spine Stenosis on MRI Using CNN},
  author  = {Shahzadi, Turrnum and Ali, Muhammad Usman and Majeed, Fiaz and
             Sana, Muhammad Usman and Diaz, Raquel Mart{\'i}nez and
             Samad, Md Abdus and Ashraf, Imran},
  journal = {Diagnostics},
  volume  = {13},
  number  = {18},
  pages   = {2975},
  year    = {2023},
  publisher = {MDPI},
  doi     = {10.3390/diagnostics13182975}
}
```

## License

Code: MIT License. See `LICENSE` for details.

Paper: Open Access under Creative Commons Attribution (CC BY 4.0).
© 2023 by the authors. Published by MDPI, Basel, Switzerland.
