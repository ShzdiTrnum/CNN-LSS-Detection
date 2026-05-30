"""
train.py
========
Training entry point for the LSS CNN classification model.

Supports three dataset configurations from the paper:
  - 5 k   : batch_size=64,  epochs=200
  - 10 k  : batch_size=128, epochs=34
  - 12.5 k: batch_size=256, epochs=34  ← best results (97.01% / 97.71%)

Usage (Colab):
    from src.train import run_training
    history, model = run_training(dataset_size='12.5k', roi_type='single')

CLI:
    python src/train.py --dataset_size 12.5k --roi_type single
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from src.data_utils import load_dataset, preprocess, split_and_encode
from src.model import build_model, compile_model
from src.evaluate import plot_history, evaluate_model


# ─────────────────────────────────────────────────────────────────────
# Dataset size configurations (per paper Section 3.2)
# ─────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    '5k':   {'batch_size': 64,  'epochs': 200},
    '10k':  {'batch_size': 128, 'epochs': 34},
    '12.5k': {'batch_size': 256, 'epochs': 34},
}

OUTPUT_DIR = '/content/drive/MyDrive/LSS_CNN_Results'


def run_training(data_path:     str   = '/content/drive/MyDrive/Orignal_Data.npz',
                 dataset_size:  str   = '12.5k',
                 roi_type:      str   = 'single',
                 output_dir:    str   = OUTPUT_DIR,
                 learning_rate: float = 0.001,
                 test_size:     float = 0.00833,
                 random_state:  int   = 1):
    """
    Full training pipeline.

    Args:
        data_path:     path to the .npz dataset archive.
        dataset_size:  '5k', '10k', or '12.5k' — controls batch size and epochs.
        roi_type:      'single' (AAP only) or 'multi' (IVD + PE + TS + AAP).
        output_dir:    directory to save model weights and plots.
        learning_rate: Adam optimizer learning rate (default 0.001).
        test_size:     fraction of data for testing (default 0.00833 ≈ 330 images).
        random_state:  random seed for reproducibility.

    Returns:
        (history, model)
    """
    print("=" * 60)
    print(f"  LSS CNN Training — {roi_type.upper()}-ROI | {dataset_size} dataset")
    print("=" * 60)

    if dataset_size not in DATASET_CONFIGS:
        raise ValueError(f"dataset_size must be one of {list(DATASET_CONFIGS.keys())}")

    config     = DATASET_CONFIGS[dataset_size]
    batch_size = config['batch_size']
    epochs     = config['epochs']
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print(f"\n[1/4] Loading dataset ...")
    (x_train, y_train), (x_test, y_test) = load_dataset(data_path)

    # 2. Preprocess
    print(f"\n[2/4] Preprocessing ...")
    x_train, x_test = preprocess(x_train, x_test, image_size=240, channels=3)

    # 3. Split and encode
    print(f"\n[3/4] Splitting and encoding labels ...")
    x_train, x_test, y_train, y_test = split_and_encode(
        x_train, x_test, y_train, y_test,
        test_size=test_size, random_state=random_state)

    # 4. Build and compile model
    print(f"\n[4/4] Building model ...")
    model = build_model(input_shape=(240, 240, 1))
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    print(f"\n── Training ─────────────────────────────────────────")
    print(f"  Dataset size : {dataset_size}")
    print(f"  ROI type     : {roi_type}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Epochs       : {epochs}")
    print(f"  LR           : {learning_rate}")
    print(f"  Output dir   : {output_dir}")

    # 5. Train
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # 6. Save model
    model_name = f"lss_cnn_{roi_type}roi_{dataset_size}.h5"
    model_path = os.path.join(output_dir, model_name)
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # 7. Plot and evaluate
    plot_history(history, output_dir=output_dir,
                 tag=f"{roi_type}roi_{dataset_size}")
    evaluate_model(model, x_test, y_test, output_dir=output_dir,
                   tag=f"{roi_type}roi_{dataset_size}")

    return history, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN for Lumbar Spine Stenosis classification")
    parser.add_argument("--data_path",    default='/content/drive/MyDrive/Orignal_Data.npz',
                        help="Path to .npz dataset file")
    parser.add_argument("--dataset_size", default='12.5k',
                        choices=['5k', '10k', '12.5k'],
                        help="Training dataset size configuration")
    parser.add_argument("--roi_type",     default='single',
                        choices=['single', 'multi'],
                        help="single-ROI (AAP) or multi-ROI (IVD+PE+TS+AAP)")
    parser.add_argument("--output_dir",   default=OUTPUT_DIR,
                        help="Directory to save model and plots")
    parser.add_argument("--lr",           type=float, default=0.001,
                        help="Adam learning rate (default 0.001)")
    args = parser.parse_args()

    run_training(
        data_path    = args.data_path,
        dataset_size = args.dataset_size,
        roi_type     = args.roi_type,
        output_dir   = args.output_dir,
        learning_rate= args.lr,
    )
