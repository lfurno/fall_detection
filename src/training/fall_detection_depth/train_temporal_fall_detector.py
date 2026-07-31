"""
Train and evaluate a temporal CNN that detects completed falls in the
UR Fall Detection Dataset using depth-camera features.

The script classifies temporal windows of consecutive frames as
"not fall" or "fall".

Dataset:
The script expects two CSV files in --data_dir:
  - urfall-cam0-falls.csv   (fall sequences)
  - urfall-cam0-adls.csv    (Activities of Daily Living sequences - ADL -)
  downloaded from https://fenix.ur.edu.pl/~mkepski/ds/uf.html.
Each row contains per-frame posture features. Labels are encoded as:
  -1: not lying
   0: transition
   1: lying

A positive temporal window represents the completion of a fall: a transition
frame (0) followed by a lying frame (1). The initial not-lying state (-1) does
not need to be visible in the same window. ADL windows remain negative.

Pipeline overview:
1. Load and concatenate the two CSVs.
2. Hold out a stratified (by fall/ADL) split of whole sequences as a test
   set, which remains untouched until final evaluation.
3. Slice the remaining ("dev") sequences into fixed-size, overlapping
   temporal windows, and label each window (see label_window()).
4. Cross-validate a TemporalCNNFallDetector with StratifiedGroupKFold
   (grouped by sequence_name, so windows from the same recording never
   appear in both train and validation).
5. Pool out-of-fold (OOF) probabilities across all folds and pick an
   operating threshold that satisfies the minimum recall while maximizing
   specificity.
6. Train the final model on all development windows for the median best
   epoch selected across folds, then evaluate it once on the test set.
7. Save checkpoints, the ONNX model, plots, split metadata, and evaluation
   results to --output_dir.


Usage:
    python train_temporal_fall_detector.py \
        --data_dir  path/to/csv_folder \
        --output_dir path/to/output_folder
"""
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import onnx_export as oe
import training_plots as tp
import config as cfg
import temporal_cnn as tcnn
import temporal_windows as tw

def set_reproducible_seed() -> None:
    """Seed NumPy and PyTorch and configure deterministic CUDA behavior."""

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_sequences(df: pd.DataFrame, output_dir: Path
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified sequence-level development/test split."""

    sequence_df = (
        df[["sequence_name"]]
        .drop_duplicates()
        .sort_values("sequence_name")
        .reset_index(drop=True)
    )

    sequence_df["sequence_type"] = (
        sequence_df["sequence_name"].map(tw.sequence_type))

    dev_sequences, test_sequences = train_test_split(
        sequence_df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.SEED,
        stratify=sequence_df["sequence_type"]
    )

    dev_sequences = (
        dev_sequences
        .sort_values("sequence_name")
        .reset_index(drop=True)
    )

    test_sequences = (
        test_sequences
        .sort_values("sequence_name")
        .reset_index(drop=True)
    )

    dev_names = set(dev_sequences["sequence_name"])
    test_names = set(test_sequences["sequence_name"])
    if dev_names.intersection(test_names):
        raise RuntimeError("Development and test sets overlap.")

    dev_df = df[df["sequence_name"].isin(dev_sequences["sequence_name"])].copy()
    test_df = df[df["sequence_name"].isin(test_sequences["sequence_name"])].copy()
    print(f"Development sequences: {len(dev_names)}")
    print(f"Held-out test sequences: {len(test_names)}")

    test_sequences_path = output_dir / "test_sequences.csv"
    test_sequences.to_csv(test_sequences_path, index=False)
    print(f"Saved held-out test sequences to : {test_sequences_path}")

    return dev_df, test_df

def get_window_arrays(windows: list[tw.WindowSample]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return target and grouping arrays used by grouped cross-validation."""

    y = np.array([sample.y for sample in windows], dtype=np.int64)
    groups = np.array([sample.sequence_name for sample in windows])

    return y, groups

def fit_window_scaler(windows: list[tw.WindowSample]) -> list[tw.WindowSample]:
    """Fit a feature scaler using frames from the training windows only."""

    if not windows:
        raise ValueError("Cannot fit a scaler on an empty window collection.")

    scaler = StandardScaler()
    frames = np.concatenate([samples.x for samples in windows], axis=0)
    scaler.fit(frames)

    return scaler

def transform_windows(windows: Sequence[tw.WindowSample], scaler: StandardScaler
                      ) -> list[tw.WindowSample]:
    """Return scaled copies without mutating the original samples."""

    transformed: list[tw.WindowSample] = []
    for sample in windows:
        x_scaled = scaler.transform(sample.x).astype(np.float32)
        transformed.append(
                tw.WindowSample(
                    sequence_name=sample.sequence_name,
                    start_frame=sample.start_frame,
                    end_frame=sample.end_frame,
                    x=x_scaled,
                    y=sample.y
                )
        )
    return transformed

def select_threshold_for_minimum_recall(
        probabilities: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray
) -> tuple[float, float, float]:
    """
    Select the threshold with the highest specificity among those satisfying
    MINIMUM_ALLOWED_RECALL. Prefer higher recall when specificity is equal,
    then prefer the highest threshold.
    """

    if not 0.0 <= cfg.MINIMUM_ALLOWED_RECALL <= 1.0:
        raise ValueError("MINIMUM_ALLOWED_RECALL must be between 0 and 1")

    probs_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)

    if probs_array.shape != labels_array.shape:
        raise ValueError(
            "probs and labels must have the same shape. Received "
            f"{probs_array.shape} and {labels_array.shape}."
        )

    if not np.all(np.isfinite(probs_array)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    false_positive_rate, recall, thresholds = roc_curve(
        labels_array,
        probs_array,
        pos_label=1,
        drop_intermediate=False
    )

    specificities = 1.0 - false_positive_rate
    valid_indices = np.flatnonzero(recall >= cfg.MINIMUM_ALLOWED_RECALL)

    if valid_indices.size == 0:
        raise ValueError(
            "No threshold achieved the required minimum recall of "
            f"{cfg.MINIMUM_ALLOWED_RECALL:.3f}"
        )

    best_index = max(
        valid_indices,
        key=lambda index: (
            specificities[index],
            recall[index],
            thresholds[index]
        )
    )

    return (float(thresholds[best_index]),
            float(recall[best_index]),
            float(specificities[best_index]))

def compute_pos_weight(
        windows: list[tw.WindowSample],
        device: torch.device) -> torch.Tensor:
    """Return the negative-to-positive ratio used to weight positive samples."""

    y = (np.array([sample.y for sample in windows], dtype=np.int64))
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Training data must contain both classes: received "
            f"{n_neg} negative and {n_pos} positive windows."
        )

    pos_weights = (
        torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device))

    return pos_weights

def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target labels and sigmoid probabilities for one data loader."""

    model.eval()
    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for X, y in data_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)

            labels.append(y.to(dtype=torch.int64).cpu())
            probabilities.append(probs.cpu())

    if not labels:
        raise ValueError("Prediction DataLoader produced no batches.")

    return (
        torch.cat(labels).numpy(),
        torch.cat(probabilities).numpy()
    )

def train_fold(
        dev_windows: list[tw.WindowSample],
        train_idx: list[int],
        val_idx: list[int],
        num_features: int,
        fold_number: int,
        num_workers: int,
        device: torch.device,
        checkpoint_dir: Path
) -> dict[str, np.any]:
    """Train one grouped CV fold and predict its validation windows."""

    train_windows = [dev_windows[i] for i in train_idx]
    val_windows = [dev_windows[i] for i in val_idx]

    scaler = fit_window_scaler(train_windows)
    scaled_train_windows = transform_windows(train_windows, scaler)
    scaled_val_windows = transform_windows(val_windows, scaler)

    train_loader = DataLoader(
        tw.TemporalWindowDataset(scaled_train_windows),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        tw.TemporalWindowDataset(scaled_val_windows),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    model = tcnn.TemporalCNNFallDetector(num_features).to(device)

    pos_weights = compute_pos_weight(scaled_train_windows, device)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    val_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Track the validation-loss minimum used for early stopping.
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    epochs_without_improvement = 0

    checkpoint_path = (
        checkpoint_dir / f"TemporalCNN_fold_{fold_number}.pth"
    )

    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(1, cfg.EPOCHS + 1):

        model.train()

        train_batch_losses: list[float] = []

        progress_bar = tqdm(
            train_loader,
            desc=(f"Fold {fold_number} / {cfg.FOLDS} "
                  f"Epoch {epoch} / {cfg.EPOCHS}"))

        for X, y in progress_bar:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = train_criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_batch_losses))
        train_history.append(train_loss)

        # Measure unweighted validation loss for model selection.
        model.eval()

        val_batch_losses: list[float] = []

        with torch.inference_mode():
            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = val_criterion(logits, y)
                val_batch_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_batch_losses))
        if not np.isfinite(val_loss):
            raise FloatingPointError(
                f"Fold {fold_number} produced a non-finite validation loss."
            )
        val_history.append(val_loss)

        if val_loss < best_val_loss - cfg.MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "validation_loss": best_val_loss,
                    "pos_weight": float(pos_weights.detach().cpu().item()),
                    "scaler_mean": scaler.mean_.astype(np.float32),
                    "scaler_scale": scaler.scale_.astype(np.float32),
                    "fold": int(fold_number),
                    "epoch": int(best_epoch)
                },
                checkpoint_path
            )
            print(
                f"Saved epoch {best_epoch} with validation loss "
                f"{best_val_loss:.4f}."
            )
        else:
            epochs_without_improvement += 1
            print(
                "No validation-loss improvement "
                f"for {epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_model_state is None:
        raise RuntimeError(
            f"Fold {fold_number} did not produce a valid checkpoint.")

    # Restore the best model for this fold
    model.load_state_dict(best_model_state)

    val_labels, val_probabilities = (
        predict_probabilities(model, val_loader, device)
    )
    if not np.all(np.isfinite(val_probabilities)):
        raise RuntimeError(
            "Some windows in the validation set did not receive predictions."
        )

    # Release GPU memory before starting the next fold.
    del model
    del optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    history = {
        "training": train_history,
        "validation": val_history
    }

    fold_results = {
        "history": history,
        "train_idx": train_idx.copy(),
        "val_idx": val_idx.copy(),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "validation_labels": val_labels,
        "validation_probabilities": val_probabilities,
        "checkpoint_path": str(checkpoint_path)
    }

    print(f"Fold {fold_number} completed. Best epoch: {best_epoch}. "
          f"Best validation loss: {best_val_loss:.4f}.")

    return fold_results

def train_final_model(
        train_windows: Sequence[tw.WindowSample],
        test_windows: Sequence[tw.WindowSample],
        epochs: int,
        threshold: float,
        num_workers: int,
        device: torch.device,
        checkpoint_dir: Path
) -> None:
    """Train the final model on development windows and predict test windows."""

    scaler = fit_window_scaler(train_windows)
    scaled_train_windows = transform_windows(train_windows, scaler)
    scaled_test_windows = transform_windows(test_windows, scaler)

    dev_loader = DataLoader(
        tw.TemporalWindowDataset(scaled_train_windows),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    test_loader = DataLoader(
        tw.TemporalWindowDataset(scaled_test_windows),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    model = tcnn.TemporalCNNFallDetector(len(cfg.FEATURES)).to(device)

    pos_weights = compute_pos_weight(scaled_train_windows, device)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    train_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []

        progress_bar = tqdm(
            dev_loader,
            desc=f"Epoch {epoch}/{cfg.EPOCHS}",
            leave=False,
        )

        for x, y in progress_bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = train_criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            batch_losses.append(loss_value)
            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        train_loss = float(np.mean(batch_losses))
        train_history.append(train_loss)

    test_labels, test_probabilities = (
        predict_probabilities(model, test_loader, device)
    )

    checkpoint_path = (
        checkpoint_dir / "final_model.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": torch.as_tensor(
                scaler.mean_,
                dtype=torch.float32
            ).cpu(),
            "scaler_scale": torch.as_tensor(
                scaler.scale_,
                dtype=torch.float32
            ).cpu(),
            "features_names": cfg.FEATURES,
            "dropout": cfg.DROPOUT,
            "kernel_size": cfg.KERNEL_SIZE,
            "padding": cfg.PADDING,
            "window_size": cfg.WINDOW_SIZE,
            "window_stride": cfg.WINDOW_STRIDE,
            "threshold": threshold,
            "num_epochs": epochs
        },
        checkpoint_path
    )

    del model
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Training final model completed.")

    return {
        "history": {"training": train_history },
        "test_labels": test_labels,
        "test_probabilities": test_probabilities
    }

def validate_cross_validation_targets(
        labels: np.ndarray, groups: np.ndarray) -> None:
    """Fail early when grouped CV cannot form meaningful positive folds."""

    if np.unique(labels).size != 2:
        raise ValueError("Development windows must contain both classes.")

    positive_groups = np.unique(groups[labels == 1])
    if len(positive_groups) < cfg.FOLDS:
        raise ValueError(
            f"Only {len(positive_groups)} sequences contain positive windows, "
            f"but FOLDS={cfg.FOLDS}. Reduce FOLDS or revisit the window label rule."
        )

def calculate_metrics(
        labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, float], np.ndarray]:
    """Calculate classification metrics, ranking metrics, and the confusion matrix."""

    predictions = (probabilities >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    has_both_classes = np.unique(labels).size == 2
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0),
        "recall": recall_score(
            labels, predictions, zero_division=0),
        "f1": f1_score(
            labels, predictions, zero_division=0),
        "specificity": float(specificity),
        "auroc": (
            roc_auc_score(labels, probabilities)
            if has_both_classes
            else float("nan")
        ),
        "average_precision": (
            average_precision_score(labels, probabilities)
            if has_both_classes
            else float("nan")
        ),
    }
    return metrics, cm

def run(args: argparse.Namespace) -> None:
    """Run cross-validation, final training, and held-out test evaluation."""

    set_reproducible_seed()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir.resolve()}")

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    if args.cuda and device.type != "cuda":
        print("CUDA was requested but is unavailable. Using CPU instead.")
    print(f"Device: {device}")

    full_df = tw.load_and_concat_dataset(data_dir)
    dev_df, test_df = split_sequences(full_df, output_dir)

    dev_windows = tw.build_temporal_window_dataset(dev_df)
    if not dev_windows:
        raise ValueError("No development windows were created.")
    test_windows = tw.build_temporal_window_dataset(test_df)
    if not test_windows:
        raise ValueError("No test windows were created.")

    dev_labels, dev_groups = get_window_arrays(dev_windows)
    validate_cross_validation_targets(dev_labels, dev_groups)

    test_labels, _ = get_window_arrays(test_windows)

    print(f"Development windows: {len(dev_windows)}")
    print(f"Positive: {int(np.sum(dev_labels == 1))}")
    print(f"Negative: {int(np.sum(dev_labels == 0))}")
    print(f"Sequences: {len(np.unique(dev_groups))}")
    print(f"Held-out test windows: {len(test_windows)}")
    print(f"Positive: {int(np.sum(test_labels == 1))}")
    print(f"Negative: {int(np.sum(test_labels == 0))}")

    tp.save_temporal_windows_distribution(
        dev_labels,
        output_dir / "development_window_distribution.png",
        "Development temporal-window distribution"
    )

    tp.save_temporal_windows_distribution(
        test_labels,
        output_dir / "test_window_distribution.png",
        "Held-out test temporal-window distribution"
    )

    cv = StratifiedGroupKFold(
        n_splits=cfg.FOLDS, shuffle=True, random_state=cfg.SEED)

    oof_probabilities = np.full(len(dev_labels), np.nan, dtype=np.float32)
    oof_labels = dev_labels.copy()

    fold_summaries: dict[int, dict[str, np.any]] = {}
    fold_best_epoch: list[int] = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(np.zeros(len(dev_labels)), dev_labels, dev_groups),
        start=1
    ):
        fold_results = train_fold(
            dev_windows=dev_windows,
            train_idx=train_idx,
            val_idx=val_idx,
            num_features=len(cfg.FEATURES),
            fold_number=fold_number,
            num_workers=cfg.NUM_WORKERS,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        oof_probabilities[val_idx] = fold_results["validation_probabilities"]
        fold_best_epoch.append(fold_results["best_epoch"])
        fold_summaries[fold_number] = fold_results

        tp.save_loss_curves(
            history=fold_results["history"],
            output_path= output_dir / f"fold_{fold_number}_loss.png",
            title=f"Fold {fold_number}",
            best_epoch=fold_results["best_epoch"],
        )

    selected_threshold, selected_recall, selected_specificity = (
        select_threshold_for_minimum_recall(
            probabilities=oof_probabilities,
            labels=oof_labels
        )
    )
    print(f"Selected threshold: {selected_threshold:.4f}.")
    print(f"Recall at selected threshold: {selected_recall:.4f}.")
    print(f"Specificity at selected threshold: {selected_specificity:.4f}.")

    tp.save_probability_distribution(
        labels=oof_labels,
        probabilities=oof_probabilities,
        threshold=selected_threshold,
        output_path=output_dir / "oof_probability_distribution.png",
        title="Development OOF fall-probability distribution"
    )

    tp.save_threshold_selection_plot(
        labels=oof_labels,
        probabilities=oof_probabilities,
        selected_threshold=selected_threshold,
        minimum_recall=cfg.MINIMUM_ALLOWED_RECALL,
        output_path=output_dir / "threshold_selection.png"
    )

    dev_metrics, dev_cm = (
        calculate_metrics(
            labels=dev_labels,
            probabilities=oof_probabilities,
            threshold=selected_threshold)
            )
    tp.print_metrics("Development out-of-fold metrics", dev_metrics)

    fold_metrics = {}
    for fold_number, fold_results in fold_summaries.items():
        metrics, _ = calculate_metrics(
            labels=fold_results["validation_labels"],
            probabilities=fold_results["validation_probabilities"],
            threshold=selected_threshold
        )
        fold_metrics[fold_number] = metrics

    fold_metrics_df = pd.DataFrame.from_dict(
        fold_metrics,
        orient="index"
    )
    mean_metrics = fold_metrics_df.mean().to_dict()
    std_metrics = fold_metrics_df.std(ddof=1).to_dict()

    tp.print_cross_validation_metrics(
        "Cross-validation metrics at selected threshold",
        {
            "mean": mean_metrics,
            "std": std_metrics
        }
    )

    tp.save_confusion_matrix(
        confusion_matrix=dev_cm,
        output_path=output_dir / "validation_confusion_matrix.png",
        title = "Validation confusion matrix"
        )

    epochs = int(np.median(fold_best_epoch))
    final_model_results = train_final_model(
        train_windows=dev_windows,
        test_windows=test_windows,
        epochs=epochs,
        threshold=selected_threshold,
        num_workers=cfg.NUM_WORKERS,
        device=device,
        checkpoint_dir=checkpoint_dir
        )

    tp.save_loss_curves(
        history=final_model_results["history"],
        output_path=output_dir / "final_model_training_loss.png",
        title="Final model training loss"
    )

    test_metrics, test_cm = calculate_metrics(
        labels=test_labels,
        probabilities=final_model_results["test_probabilities"],
        threshold=selected_threshold
    )
    tp.print_metrics("Held-out test metrics", test_metrics)

    tp.save_confusion_matrix(
        confusion_matrix=test_cm,
        output_path=output_dir / "test_confusion_matrix.png",
        title = "Held-out test confusion matrix"
        )

    results = {
        "config": {
            "seed": cfg.SEED,
            "test_size": cfg.TEST_SIZE,
            "window_size": cfg.WINDOW_SIZE,
            "window_stride": cfg.WINDOW_STRIDE,
            "features": cfg.FEATURES,
            "kernel_size": cfg.KERNEL_SIZE,
            "padding": cfg.PADDING,
            "dropout": cfg.DROPOUT,
            "num_folds": cfg.FOLDS,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "patience": cfg.PATIENCE,
            "min_delta": cfg.MIN_DELTA,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "minimum_allowed_recall": cfg.MINIMUM_ALLOWED_RECALL
        },
        "folds": fold_summaries,
        "cross_validation": {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
        },
        "development_oof": {
            "threshold": selected_threshold,
            "recall_at_threshold_selection": selected_recall,
            "specificity_at_threshold_selection": selected_specificity,
            "metrics": dev_metrics,
            "confusion_matrix": dev_cm.tolist(),
        },
        "held_out_test": {
            "metrics": test_metrics,
            "confusion_matrix": test_cm.tolist(),
        }
    }

    torch.save(results, output_dir / "experiment_results.pth")

    oe.export_final_model_to_onnx(
        checkpoint_path= checkpoint_dir / "final_model.pth",
        output_path=models_dir / "final_model.onnx"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Train and evaluate a temporal CNN fall detector "
                                     "using depth-camera features.")
    parser.add_argument("--data_dir",
                        default='../../../share/features_from_depth_camera',
                        help="Directory containing urfall-cam0-falls.csv "
                             "and urfall-cam0-adls.csv.")
    parser.add_argument("--output_dir",
                        default='output/fall_detector_depth',
                        help="Directory for checkpoints, the ONNX model, plots, "
                             "split metadata, and evaluation results.")
    parser.add_argument("--cuda",
                        action="store_true",
                        help="Use CUDA when available. Otherwise, use CPU.")

    args = parser.parse_args()

    run(args)
