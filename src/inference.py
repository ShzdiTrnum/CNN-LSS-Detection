"""
inference.py
============
Load a trained model and run inference on new MRI images.

Usage (Colab):
    from src.inference import load_model, predict_image, predict_folder
    model = load_model('/content/drive/MyDrive/LSS_CNN_Results/lss_cnn_singleroi_12.5k.h5')
    label, confidence = predict_image(model, '/path/to/image.jpg')
"""

import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model


CLASS_NAMES  = ['Normal', 'Mild', 'Moderate', 'Severe']
IMAGE_SIZE   = 240


def load_model(model_path: str):
    """
    Load a trained Keras model from disk.

    Args:
        model_path: path to the saved .h5 model file.

    Returns:
        Loaded Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras_load_model(model_path)
    print(f"Model loaded: {model_path}")
    return model


def preprocess_image(img_path: str,
                     image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Load, resize, normalise, and expand dims for a single MRI image.

    Args:
        img_path:   path to the image file.
        image_size: target spatial size (default 240).

    Returns:
        Preprocessed array of shape (1, image_size, image_size, 1).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))   # (1, H, W, 1)
    return img


def predict_image(model, img_path: str) -> tuple:
    """
    Predict the LSS grade for a single MRI image.

    Args:
        model:    trained Keras model.
        img_path: path to the MRI image.

    Returns:
        (label, confidence) — predicted class name and probability.
    """
    img        = preprocess_image(img_path)
    probs      = model.predict(img, verbose=0)[0]
    class_idx  = int(np.argmax(probs))
    label      = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])

    print(f"Image   : {os.path.basename(img_path)}")
    print(f"Grade   : {label}  (confidence: {confidence:.4f})")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        marker = " ←" if i == class_idx else ""
        print(f"  {cls:<10}: {p:.4f}{marker}")

    return label, confidence


def predict_folder(model, folder_path: str,
                   img_ext: str = '*.jpg') -> list:
    """
    Run inference on all images in a folder.

    Args:
        model:       trained Keras model.
        folder_path: path to folder containing MRI images.
        img_ext:     glob pattern for images.

    Returns:
        List of (filename, label, confidence) tuples.
    """
    from pathlib import Path
    results = []
    images  = list(Path(folder_path).glob(img_ext))

    if not images:
        print(f"No images found in {folder_path} matching {img_ext}")
        return results

    print(f"\nRunning inference on {len(images)} images ...")
    for img_path in sorted(images):
        try:
            label, conf = predict_image(model, str(img_path))
            results.append((img_path.name, label, conf))
        except Exception as e:
            print(f"  Skipped {img_path.name}: {e}")

    # Summary
    from collections import Counter
    counts = Counter(r[1] for r in results)
    print(f"\n── Summary ───────────────────────────────────────────")
    for cls in CLASS_NAMES:
        print(f"  {cls:<10}: {counts.get(cls, 0)} images")

    return results
