"""Plot and print temporal fall-detector training results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

# For binary classification, the class names are "NOT FALL" and "FALL".
CLASS_NAMES = ["NOT FALL", "FALL"]

def save_temporal_windows_distribution(
        labels: pd.Series, output_path: Path, title: str
        ) -> None:
    """Save a class-balance bar chart for temporal windows."""

    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([0, 1], counts, color = ["green", "red"])
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_ylabel("Number of windows")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_probability_distribution(
        labels: np.ndarray,
        probabilities: np.ndarray,
        threshold: float,
        output_path: Path,
        title: str
        ) -> None:
    """Save fall-probability histograms for negative and positive windows."""

    bins = np.linspace(0.0, 1.0, 21)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        probabilities[labels == 0],
        bins=bins,
        color="green",
        alpha=0.65,
        label=CLASS_NAMES[0]
    )
    ax.hist(
        probabilities[labels == 1],
        bins=bins,
        color="red",
        alpha=0.65,
        label=CLASS_NAMES[1]
    )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"Threshold ({threshold:.3f})"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted fall probability")
    ax.set_ylabel("Number of windows")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_loss_curves(
    history: dict[str, list[float]], output_path: Path,
    title: str, best_epoch: int | None = None) -> None:
    """Save the available training and validation loss curves."""

    fig, ax = plt.subplots(figsize=(8, 4))

    for split_name, losses in history.items():
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, label=split_name.capitalize())

    if best_epoch is not None:
        validation_losses = history["validation"]
        ax.plot(best_epoch, validation_losses[best_epoch - 1],
                marker='*', markersize=13, linestyle="none", color="gold",
                markeredgecolor='black', label="Best Epoch")
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_threshold_selection_plot(
        labels: np.ndarray,
        probabilities: np.ndarray,
        selected_threshold: float,
        minimum_recall: float,
        output_path: Path
        ) -> None:
    """Plot the recall-specificity trade-off used to select the threshold."""

    false_positive_rate, recall, thresholds = roc_curve(
        labels,
        probabilities,
        pos_label=1,
        drop_intermediate=False
    )
    specificity = 1.0 - false_positive_rate

    finite_thresholds = np.isfinite(thresholds)
    thresholds = thresholds[finite_thresholds]
    recall = recall[finite_thresholds]
    specificity = specificity[finite_thresholds]

    # Display thresholds from low to high: recall generally decreases while
    # specificity increases as the decision rule becomes more restrictive.
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    recall = recall[order]
    specificity = specificity[order]

    selected_index = int(
        np.argmin(np.abs(thresholds - selected_threshold))
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(thresholds, recall, where="post", color="red", label="Recall")
    ax.step(
        thresholds,
        specificity,
        where="post",
        color="blue",
        label="Specificity"
    )
    ax.axhline(
        minimum_recall,
        color="red",
        linestyle=":",
        label=f"Minimum recall ({minimum_recall:.2f})"
    )
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected threshold ({selected_threshold:.3f})"
    )
    ax.scatter(
        selected_threshold,
        recall[selected_index],
        color="red",
        zorder=3
    )
    ax.scatter(
        selected_threshold,
        specificity[selected_index],
        color="blue",
        zorder=3
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Decision threshold", fontsize=14)
    ax.set_ylabel("Metric value", fontsize=14)
    ax.set_title("OOF threshold selection", fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_confusion_matrix(
        confusion_matrix: np.ndarray, output_path: Path, title: str
        ) -> None :
    """Save a labeled confusion-matrix plot."""

    fig, ax = plt.subplots(figsize=(11, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=CLASS_NAMES
    )

    display.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    for text in display.text_.ravel():
        text.set_fontsize(16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("True", fontsize=16)
    ax.set_title(title, fontsize=18)
    fig.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Print metrics as a compact one-row table."""

    print(f"\n{title}\n")
    metrics_df = pd.DataFrame(metrics, index=[0]).round(4)
    print(metrics_df.to_string(index=False))

def print_cross_validation_metrics(
        title: str,
        metrics: dict[str, dict[str, float]]
        ) -> None:
    """Print metric summaries as a compact table."""

    print(f"\n{title}\n")
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").round(4)
    print(metrics_df.to_string())
