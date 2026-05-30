"""
evaluate.py
===========
Evaluation, plotting, and metrics for the LSS CNN model.

Generates:
  - Training/validation accuracy and loss curves
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix
  - Augmented vs non-augmented accuracy comparison bar chart

Usage:
    from src.evaluate import plot_history, evaluate_model, plot_augmentation_comparison
    plot_history(history, output_dir, tag='single_12.5k')
    evaluate_model(model, x_test, y_test, output_dir, tag='single_12.5k')
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


CLASS_NAMES = ['Normal', 'Mild', 'Moderate', 'Severe']


# ─────────────────────────────────────────────────────────────────────
# Training history plots
# ─────────────────────────────────────────────────────────────────────

def plot_history(history, output_dir: str, tag: str = ''):
    """
    Plot and save training/validation accuracy and loss curves.

    Produces two figures (accuracy and loss), each saved as PNG.

    Args:
        history:    Keras History object from model.fit().
        output_dir: directory to save plots.
        tag:        filename suffix (e.g. 'single_12.5k').
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history.history['accuracy']) + 1)

    # ── Accuracy plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history['accuracy'],
            color='#2980B9', lw=2, label='Training acc')
    ax.plot(epochs, history.history['val_accuracy'],
            color='#E74C3C', lw=2, ls='--', label='Validation acc')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'accuracy_{tag}.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Loss plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history['loss'],
            color='#E74C3C', lw=2, label='Training loss')
    ax.plot(epochs, history.history['val_loss'],
            color='#2980B9', lw=2, ls='--', label='Validation loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'loss_{tag}.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Model evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(model, x_test: np.ndarray, y_test_enc: np.ndarray,
                   output_dir: str, tag: str = ''):
    """
    Evaluate model on test data and print/save classification report
    and confusion matrix.

    Args:
        model:       trained Keras model.
        x_test:      test images.
        y_test_enc:  one-hot encoded test labels.
        output_dir:  directory to save results.
        tag:         filename suffix.

    Returns:
        Dict with test_loss, test_accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Test loss and accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test_enc, verbose=0)
    print(f"\n── Test Results ({tag}) ───────────────────────────────")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # Predictions
    y_pred  = model.predict(x_test, verbose=0)
    y_pred  = np.argmax(y_pred, axis=1)
    y_true  = np.argmax(y_test_enc, axis=1)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=2)
    print(f"\n── Classification Report ─────────────────────────────")
    print(report)

    report_path = os.path.join(output_dir, f'classification_report_{tag}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        f.write(report)
    print(f"Saved: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Confusion Matrix ({tag})', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_{tag}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_path}")

    return {'test_loss': test_loss, 'test_accuracy': test_acc}


# ─────────────────────────────────────────────────────────────────────
# Dataset size comparison plot (Figure 7 equivalent)
# ─────────────────────────────────────────────────────────────────────

def plot_dataset_size_comparison(results: dict, output_dir: str):
    """
    Bar chart comparing accuracy across different dataset sizes
    for Multi-ROI and Single-ROI (reproduces Figure 7 in the paper).

    Args:
        results: dict with structure:
            {
              'multi': {'5k': acc, '10k': acc, '12.5k': acc},
              'single': {'5k': acc, '10k': acc, '12.5k': acc},
            }
        output_dir: directory to save the chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    sizes    = ['5K', '10K', '12.5K']
    multi_acc  = [results['multi'].get('5k', 0),
                  results['multi'].get('10k', 0),
                  results['multi'].get('12.5k', 0)]
    single_acc = [results['single'].get('5k', 0),
                  results['single'].get('10k', 0),
                  results['single'].get('12.5k', 0)]

    x   = np.arange(len(sizes))
    w   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Classification Accuracy across Dataset Sizes',
                 fontsize=13, fontweight='bold')

    for ax, acc, title, color in [
        (axes[0], multi_acc,  'Multi-ROI',  '#2980B9'),
        (axes[1], single_acc, 'Single-ROI', '#27AE60'),
    ]:
        bars = ax.bar(x, [a * 100 for a in acc], color=color,
                      alpha=0.85, width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(sizes, fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, axis='y', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'dataset_size_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Augmented vs non-augmented comparison (Figure 8 equivalent)
# ─────────────────────────────────────────────────────────────────────

def plot_augmentation_comparison(output_dir: str):
    """
    Bar chart comparing augmented vs non-augmented accuracy
    for Multi-ROI and Single-ROI (reproduces Figure 8 in the paper).

    Values are hardcoded from Table 6 of the paper.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Table 6 values
    data = {
        'Non-augmented (1.5k)': {'multi': 36.47, 'single': 35.66},
        'Augmented 5k':         {'multi': 86.30, 'single': 93.15},
        'Augmented 10k':        {'multi': 95.83, 'single': 97.71},
        'Augmented 12.5k':      {'multi': 97.01, 'single': 97.71},
    }

    labels      = list(data.keys())
    multi_vals  = [data[l]['multi']  for l in labels]
    single_vals = [data[l]['single'] for l in labels]
    x  = np.arange(len(labels))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, multi_vals,  w, label='Multi-ROI',
                   color='#2980B9', alpha=0.85)
    bars2 = ax.bar(x + w/2, single_vals, w, label='Single-ROI',
                   color='#27AE60', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Augmented vs Non-Augmented Dataset Accuracy\n'
                 '(Multi-ROI vs Single-ROI)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'augmented_vs_nonaug.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# State-of-the-art comparison (Table 7 equivalent)
# ─────────────────────────────────────────────────────────────────────

def plot_sota_comparison(output_dir: str):
    """
    Horizontal bar chart comparing the proposed CNN against existing
    state-of-the-art methods (Table 7 from the paper).
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = [
        ('CNN [Salehi et al.]',        87.75),
        ('U-Net [Lu et al.]',           94.00),
        ('DMML-Net [Han et al.]',       84.50),
        ('CNN [Hallinan et al.]',        84.50),
        ('VGG16 [Altun et al.]',        87.70),
        ('Different models [Fujiwara]', 95.00),
        ('Proposed CNN\n(Single-ROI 12.5k)', 97.71),
    ]
    labels  = [m[0] for m in methods]
    accs    = [m[1] for m in methods]
    colors  = ['#95A5A6'] * (len(methods) - 1) + ['#E74C3C']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, accs, color=colors, alpha=0.88)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Performance Comparison with State-of-the-Art Methods',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(70, 102)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.2, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%',
                va='center', ha='left', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'sota_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
