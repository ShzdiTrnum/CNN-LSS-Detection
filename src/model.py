"""
model.py
========
Custom CNN architecture for Lumbar Spine Stenosis (LSS) classification.

Architecture (per Table 1 of the paper):
  5 × (Conv2D 3×3 + ReLU + MaxPool 2×2 + Dropout 0.14)
  Filter sizes: 32 → 64 → 128 → 256 → 512
  Followed by 4 Fully Connected layers (256, 256, 256, 128)
  Output: 4-class Softmax (Normal / Mild / Moderate / Severe)

Reference:
    Shahzadi, T. et al. (2023). Nerve Root Compression Analysis to Find
    Lumbar Spine Stenosis on MRI Using CNN. Diagnostics, 13, 2975.
    https://doi.org/10.3390/diagnostics13182975

Usage:
    from src.model import build_model
    model = build_model(input_shape=(240, 240, 1))
    model.summary()
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization,
)


def build_model(input_shape: tuple = (240, 240, 1),
                num_classes: int = 4,
                dropout_rate: float = 0.14) -> Sequential:
    """
    Build and return the custom CNN model for LSS classification.

    The architecture consists of:
      - 5 convolutional blocks (Conv + ReLU + MaxPool + Dropout)
        with filter sizes 32, 64, 128, 256, 512
      - 4 fully connected layers (256, 256, 256, 128) with Dropout
      - Output layer: Dense(num_classes, softmax)

    Args:
        input_shape:  shape of each input image (H, W, C).
                      Default (240, 240, 1) for single-channel grayscale.
        num_classes:  number of output classes (4: Normal/Mild/Moderate/Severe).
        dropout_rate: dropout probability applied after each block (default 0.14).

    Returns:
        Uncompiled Keras Sequential model.
    """
    model = Sequential(name="LSS_CNN")

    # ── Convolutional block 1 ─────────────────────────────────────────
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', input_shape=input_shape,
                     name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Dropout(dropout_rate, name='drop1'))

    # ── Convolutional block 2 ─────────────────────────────────────────
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(dropout_rate, name='drop2'))

    # ── Convolutional block 3 ─────────────────────────────────────────
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', name='conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(Dropout(dropout_rate, name='drop3'))

    # ── Convolutional block 4 ─────────────────────────────────────────
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))
    model.add(Dropout(dropout_rate, name='drop4'))

    # ── Convolutional block 5 ─────────────────────────────────────────
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', name='conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool5'))
    model.add(Dropout(dropout_rate, name='drop5'))

    # ── Flatten ───────────────────────────────────────────────────────
    model.add(Flatten(name='flatten'))

    # ── Fully connected layers ────────────────────────────────────────
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dropout(dropout_rate, name='drop_fc1'))

    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dropout(dropout_rate, name='drop_fc2'))

    model.add(Dense(256, activation='relu', name='fc3'))
    model.add(Dropout(dropout_rate, name='drop_fc3'))

    model.add(Dense(128, activation='relu', name='fc4'))
    model.add(Dropout(dropout_rate, name='drop_fc4'))

    # ── Output layer ──────────────────────────────────────────────────
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model


def compile_model(model: Sequential,
                  learning_rate: float = 0.001) -> Sequential:
    """
    Compile the model with Adam optimizer and categorical cross-entropy loss.

    Args:
        model:         Keras model to compile.
        learning_rate: Adam optimizer learning rate (default 0.001).

    Returns:
        Compiled model.
    """
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
