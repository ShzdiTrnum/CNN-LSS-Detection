# CNN-LSS-Detection

> Nerve Root Compression Analysis to Find Lumbar Spine Stenosis on MRI Using CNN, a custom convolutional neural network for automated four-class grading of foraminal lumbar spine stenosis (Normal / Mild / Moderate / Severe) from axial-view MRI images.

---

## Paper

**Title:** Nerve Root Compression Analysis to Find Lumbar Spine Stenosis on MRI Using CNN

**Authors:** Turrnum Shahzadi¹, Muhammad Usman Ali², Fiaz Majeed¹, Muhammad Usman Sana¹, Raquel Martínez Diaz³˒⁴˒⁵, Md Abdus Samad⁶, Imran Ashraf⁶

**Affiliations:**
- ¹ Department of Information Technology, University of Gujrat, Gujrat 50700, Pakistan
- ² Department of Computer Science, University of Gujrat, Gujrat 50700, Pakistan
- ³ Universidad Europea del Atlántico, Santander, Spain
- ⁶ Department of Information and Communication Engineering, Yeungnam University, Republic of Korea

**Published:** *Diagnostics* **2023**, 13, 2975. MDPI (Open Access — CC BY 4.0)
**DOI:** https://doi.org/10.3390/diagnostics13182975

---

## Table of Contents

- [Project Description](#project-description)
- [Key Results](#key-results)
- [Clinical Background](#clinical-background)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Region of Interest (ROI) Types](#region-of-interest-roi-types)
- [Outputs](#outputs)
- [Comparison with State-of-the-Art](#comparison-with-state-of-the-art)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Project Description

This repository implements a custom CNN model for automated classification of **foraminal lumbar spine stenosis (LSS)** from axial-view MRI images into four severity grades: Normal, Mild, Moderate, and Severe.

The pipeline extracts four Regions of Interest (ROIs) — Intervertebral Disc (IVD), Posterior Element (PE), Thecal Sac (TS), and Anteroposterior Diameter (AAP) — and trains the CNN on both Multi-ROI (all four regions) and Single-ROI (AAP only) datasets. Data augmentation (translation, zoom, rotation) is applied to address the limited size of medical imaging datasets.

---

## Key Results

| Dataset Size | ROI Type | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|----------|-----------|--------|----------|
| 12.5k | Single-ROI | **97.71%** | 0.98 | 0.98 | 0.98 |
| 12.5k | Multi-ROI | **97.01%** | 0.97 | 0.97 | 0.97 |
| 10k | Single-ROI | 97.71% | 0.98 | 0.98 | 0.98 |
| 10k | Multi-ROI | 95.83% | 0.96 | 0.96 | 0.96 |
| 5k | Single-ROI | 93.15% | 0.92 | 0.92 | 0.92 |
| 5k | Multi-ROI | 86.30% | 0.86 | 0.85 | 0.85 |
| Non-augmented (1545) | Multi-ROI | 36.47% | — | — | — |

Best model: **Single-ROI, 12.5k augmented dataset, 97.71% accuracy**

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

The four ROIs evaluated:

| ID | ROI | Description |
|----|-----|-------------|
| 1 | IVD | Intervertebral Disc |
| 2 | PE | Posterior Element |
| 3 | TS | Thecal Sac |
| 4 | AAP | Anteroposterior Diameter (primary for foraminal stenosis) |

---

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
├── outputs/
│   └── README.md             # Output file descriptions
│
├── docs/
│   └── architecture_notes.md # CNN architecture details and design rationale
│
├── assets/                   # Sample result images
│
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

---

## Usage

### Full training pipeline (Colab)

```python
from src.train import run_training

# Best configuration from the paper: Single-ROI, 12.5k dataset
history, model = run_training(
    data_path    = '/content/drive/MyDrive/Orignal_Data.npz',
    dataset_size = '12.5k',    # '5k', '10k', or '12.5k'
    roi_type     = 'single',   # 'single' (AAP) or 'multi' (IVD+PE+TS+AAP)
    output_dir   = '/content/drive/MyDrive/LSS_CNN_Results',
)
```

### Generate comparison plots

```python
from src.evaluate import plot_augmentation_comparison, plot_sota_comparison

plot_augmentation_comparison(output_dir='/content/drive/MyDrive/LSS_CNN_Results')
plot_sota_comparison(output_dir='/content/drive/MyDrive/LSS_CNN_Results')
```

### Inference on new images

```python
from src.inference import load_model, predict_image, predict_folder

model = load_model('/content/drive/MyDrive/LSS_CNN_Results/lss_cnn_singleroi_12.5k.h5')
label, confidence = predict_image(model, '/path/to/axial_mri.jpg')
results = predict_folder(model, '/path/to/mri_folder/', img_ext='*.jpg')
```

### CLI training

```bash
python src/train.py \
    --data_path    /content/drive/MyDrive/Orignal_Data.npz \
    --dataset_size 12.5k \
    --roi_type     single \
    --output_dir   /content/drive/MyDrive/LSS_CNN_Results
```

---

## Configuration

Dataset size configurations (per paper Section 3.2):

| Dataset Size | Batch Size | Epochs | Best Use |
|-------------|-----------|--------|---------|
| `5k` | 64 | 200 | Initial experiments |
| `10k` | 128 | 34 | Intermediate results |
| `12.5k` | 256 | 34 | **Best results** (paper configuration) |

Common hyperparameters (all configurations):
- Optimizer: Adam (lr = 0.001)
- Loss: Categorical cross-entropy
- Dropout rate: 0.14 (applied after every conv block and FC layer)
- Train/test split: 80:20
- Fivefold cross-validation
- Input size: 240 × 240 × 1 (grayscale, single channel)

---

## Model Architecture

Custom CNN with 5 convolutional blocks followed by 4 fully connected layers (Table 1 of the paper):

```
Input: (240, 240, 1)
│
├── Conv2D(32, 3×3, stride=1, padding=same, ReLU)
│   MaxPool(2×2)  →  Dropout(0.14)
│
├── Conv2D(64, 3×3, stride=1, padding=same, ReLU)
│   MaxPool(2×2)  →  Dropout(0.14)
│
├── Conv2D(128, 3×3, stride=1, padding=same, ReLU)
│   MaxPool(2×2)  →  Dropout(0.14)
│
├── Conv2D(256, 3×3, stride=1, padding=same, ReLU)
│   MaxPool(2×2)  →  Dropout(0.14)
│
├── Conv2D(512, 3×3, stride=1, padding=same, ReLU)
│   MaxPool(2×2)  →  Dropout(0.14)
│
├── Flatten
│
├── Dense(256, ReLU)  →  Dropout(0.14)
├── Dense(256, ReLU)  →  Dropout(0.14)
├── Dense(256, ReLU)  →  Dropout(0.14)
├── Dense(128, ReLU)  →  Dropout(0.14)
│
└── Dense(4, Softmax)  →  [Normal, Mild, Moderate, Severe]
```

Total parameters: ~17M

---

## Region of Interest (ROI) Types

### Multi-ROI Dataset
Contains four anatomical regions extracted from axial MRI:
- **IVD** (Intervertebral Disc): structure between vertebral bodies
- **PE** (Posterior Element): Y-shaped bony structure
- **TS** (Thecal Sac): fluid-filled sac surrounding the spinal cord
- **AAP** (Anteroposterior Diameter): primary diagnostic region for foraminal stenosis

### Single-ROI Dataset
Contains only the **AAP region** — the anteroposterior diameter of the spinal canal. Clinicians evaluate three distances in the AAP to diagnose LSS: the AAP diameter and left/right foramen widths. Single-ROI consistently outperforms Multi-ROI across all dataset sizes.

---

## Outputs

All outputs saved to `output_dir`:

| File | Description |
|------|-------------|
| `lss_cnn_{roi}_{size}.h5` | Trained model weights |
| `accuracy_{roi}_{size}.png` | Training/validation accuracy curves |
| `loss_{roi}_{size}.png` | Training/validation loss curves |
| `classification_report_{roi}_{size}.txt` | Per-class precision, recall, F1 |
| `confusion_matrix_{roi}_{size}.png` | Confusion matrix heatmap |
| `dataset_size_comparison.png` | Accuracy vs dataset size bar chart |
| `augmented_vs_nonaug.png` | Augmented vs non-augmented comparison |
| `sota_comparison.png` | Comparison with state-of-the-art methods |

---

## Comparison with State-of-the-Art

| Method | Model | Accuracy |
|--------|-------|---------|
| Salehi et al. (2019) | CNN | 87.75% |
| Lu et al. (2018) | U-Net | 94.00% |
| Han et al. (2018) | DMML-Net | 84.50% |
| Hallinan et al. (2021) | CNN | 84.50% |
| Altun et al. (2023) | VGG16 | 87.70% |
| Fujiwara et al. (2023) | Various | 95.00% |
| **Proposed CNN (Single-ROI 12.5k)** | **Custom CNN** | **97.71%** |
| **Proposed CNN (Multi-ROI 12.5k)** | **Custom CNN** | **97.01%** |

---

## Known Limitations

- The model takes grayscale input (`channels=1`) but the `.npz` data loads with `channels=3` — ensure preprocessing matches your data format
- Paths are configured for Google Colab/Drive — update `DATA_PATH` and `OUTPUT_DIR` for local use
- The dataset is not included — download from Mendeley Data (see `data/README.md`)
- `from keras.layers.normalization import BatchNormalization` and `from keras.layers.advanced_activations import LeakyReLU` are legacy Keras imports — updated to TF2-compatible imports in this repository
- Non-augmented accuracy (~36%) confirms that augmentation is essential; the model cannot generalise from 1545 raw images alone

---

## Future Work

- [ ] Replace deprecated Keras layer imports with TF2-compatible equivalents
- [ ] Add transfer learning experiments (ResNet, VGG, EfficientNet)
- [ ] Add Grad-CAM visualisations for interpretability
- [ ] Support DICOM format input directly
- [ ] Add fivefold cross-validation runner
- [ ] Colab notebook (`.ipynb`) with step-by-step walkthrough

---

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

---

## License

Code: MIT License. See `LICENSE` for details.

Paper: Open Access under Creative Commons Attribution (CC BY 4.0).
© 2023 by the authors. Published by MDPI, Basel, Switzerland.
