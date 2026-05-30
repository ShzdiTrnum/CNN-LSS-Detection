"""
data_utils.py
=============
Data loading, preprocessing, and augmentation utilities for the
LSS CNN classification pipeline.

Dataset:
    Sudirman et al. (2019). Lumbar Spine MRI Dataset. Mendeley Data.
    515 patients, axial-view MRI images (L3-D3, L4-D4, L5-D5),
    labelled as Normal / Mild / Moderate / Severe by expert radiologists.

Usage:
    from src.data_utils import load_dataset, preprocess, split_dataset
    (x_train, y_train), (x_test, y_test) = load_dataset()
    x_train, x_test = preprocess(x_train, x_test)
    x_train, x_val, y_train, y_val = split_dataset(x_train, y_train)
"""

import numpy as np
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

IMAGE_SIZE  = 240    # images are cropped to 240 × 240 px (from 320 × 320)
NUM_CLASSES = 4      # Normal, Mild, Moderate, Severe
DATA_PATH   = '/content/drive/MyDrive/Orignal_Data.npz'


# ─────────────────────────────────────────────────────────────────────
# Image folder loader (optional utility for custom image folders)
# ─────────────────────────────────────────────────────────────────────

def load_image_folder(img_path: str, img_ext: str = "*.jpg") -> list:
    """
    Load all images from a folder as grayscale numpy arrays.

    Args:
        img_path: path to the image folder.
        img_ext:  glob pattern for image files (e.g. '*.jpg', '*.png').

    Returns:
        List of grayscale float32 images.
    """
    path   = Path(img_path)
    images = []
    for image_path in path.glob(img_ext):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
    print(f"Loaded {len(images)} images from {path}")
    return images


# ─────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────

def load_dataset(data_path: str = DATA_PATH):
    """
    Load the pre-processed LSS dataset from a .npz archive.

    The archive contains:
        x_train, x_test : MRI image arrays
        y_train, y_test : integer labels (0=Normal, 1=Mild, 2=Moderate, 3=Severe)

    Args:
        data_path: path to the .npz data file.

    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    data    = np.load(data_path)
    x_train = data['x_train']
    x_test  = data['x_test']
    y_train = data['y_train']
    y_test  = data['y_test']

    print(f"Loaded dataset from: {data_path}")
    print(f"  x_train : {x_train.shape}  y_train : {y_train.shape}")
    print(f"  x_test  : {x_test.shape}   y_test  : {y_test.shape}")
    print(f"  Classes : Normal=0, Mild=1, Moderate=2, Severe=3")
    return (x_train, y_train), (x_test, y_test)


# ─────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────

def preprocess(x_train: np.ndarray,
               x_test: np.ndarray,
               image_size: int = IMAGE_SIZE,
               channels: int = 3) -> tuple:
    """
    Reshape, cast to float32, and normalise images to [0, 1].

    Args:
        x_train:    raw training images.
        x_test:     raw test images.
        image_size: spatial dimension after reshape (default 240).
        channels:   number of channels (3 for RGB-format data).

    Returns:
        (x_train, x_test) as normalised float32 arrays.
    """
    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, channels)
    x_test  = x_test.reshape(x_test.shape[0],   image_size, image_size, channels)
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0
    print(f"Preprocessed: x_train={x_train.shape}  x_test={x_test.shape}")
    return x_train, x_test


# ─────────────────────────────────────────────────────────────────────
# Train / validation split and one-hot encoding
# ─────────────────────────────────────────────────────────────────────

def split_and_encode(x_train: np.ndarray, x_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     test_size: float = 0.00833,
                     random_state: int = 1,
                     num_classes: int = NUM_CLASSES) -> tuple:
    """
    Concatenate train and test, re-split with a fixed ratio, and
    one-hot encode the labels.

    The paper uses an 80:20 training/testing split (test_size ≈ 0.00833
    corresponds to ~330 test images from the 12.5 k augmented dataset).

    Args:
        x_train, x_test:   preprocessed image arrays.
        y_train, y_test:   integer label arrays.
        test_size:         fraction of data reserved for testing.
        random_state:      random seed for reproducibility.
        num_classes:       number of output classes (default 4).

    Returns:
        x_train, x_test, y_train_enc, y_test_enc
    """
    X = np.concatenate([x_train, x_test])
    Y = np.concatenate([y_train, y_test])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)

    y_train_enc = to_categorical(y_train, num_classes)
    y_test_enc  = to_categorical(y_test,  num_classes)

    print(f"\nTrain/test split (test_size={test_size}):")
    print(f"  x_train : {x_train.shape}   y_train : {y_train_enc.shape}")
    print(f"  x_test  : {x_test.shape}    y_test  : {y_test_enc.shape}")

    return x_train, x_test, y_train_enc, y_test_enc
