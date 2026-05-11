"""
Train and evaluate posture detection classifiers on the UR Fall Detection
Dataset (depth-camera features).

The current script classifies "not lying person" vs "lying person" postures, using
per frame depth features. As next step, a temporal window will be used to detect the 
"falling" state.

Dataset:
The script expects two CSV files in --data_dir:
  - urfall-cam0-falls.csv   (fall sequences)
  - urfall-cam0-adls.csv    (Activities of Daily Living sequences - ADL -) downloaded
  from https://fenix.ur.edu.pl/~mkepski/ds/uf.html.
Each row contains per frame features. Labels are encoded as:
  -1: not lying person
   0: temporary pose during "falling" (intentional or not)
   1: lying person

Pipeline overview:
1. Load and concatenate the two CSVs.
2. Preprocess: drop frames associated with temporary poses (label 0), and remap labels
   to {0: not lying, 1: lying}.
3. Save feature distributions and class balance plots to --output_dir.
4. Cross-validate three classifiers (Logistic Regression, Random Forest,
   HistGradientBoosting) with StratifiedGroupKFold).
5. Save per fold bar charts and confusion-matrix grids to --output_dir.

Usage:
    python train_frame_posture_baselines.py \
        --data_dir  path/to/csv_folder \
        --output_dir path/to/output_folder
"""

import pathlib
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Column names for the raw CSV files (no header row in the original files)
COLUMN_NAMES = [
    "sequence_name",        # Unique identifier for each video (sequence)
    "frame_number",         # Frame index within the sequence
    "label",                # Ground-truth label
    "HeightWidthRatio",     # Bounding box height to width ratio
    "MajorMinorRatio",      # Major to minor axis ratio of the fitted ellipse
    "BoundingBoxOccupancy", # Ratio of how bounding box is occupied by the silhouette
    "MaxStdXZ",             # Standard deviation of pixels from X and Z axes
    "HHmaxRatio",           # Human height in frame to human height while standing ratio
    "H",                    # Actual height (in mm)
    "D",                    # Distance of person center to the floor (in mm)
    "P40"                   # Ratio of the number of the point clouds belonging to the
                            # cuboid of 40 cm height and placed on the floor to the number
                            # of the point clouds belonging to the cuboid of height equal
                            # to person's height.
]

# Feature columns used as model inputs
FEATURES = [
    "HeightWidthRatio",
    "MajorMinorRatio",
    "BoundingBoxOccupancy",
    "MaxStdXZ",
    "HHmaxRatio",
    "H",
    "D",
    "P40"
]

MODELS = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"))
        ]),
        "RandomForest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight="balanced"))
        ]),
        "HistGradientBoosting": Pipeline([
            ("classifier", HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=10,
                learning_rate=0.01,
                random_state=42,
                class_weight="balanced"))
        ]),
    }

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_MAPPING = {-1: 0, 1: 1}

# For plotting purposes
DISPLAY_LABELS = ["Not lying", "Lying"]
METRIC_STYLES = [
    ("precision", "blue"),
    ("recall",    "red"),
    ("f1",        "green"),
]

def load_and_concat_dataset(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load and combine fall and ADL feature CSV files."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] data_dir not exists: {data_dir}")

    csv_paths = {
        'falls' : data_dir / "urfall-cam0-falls.csv",
        'adls'  : data_dir / "urfall-cam0-adls.csv",
    }

    frames: list[pd.DataFrame] = []
    for dataset_name, csv_path in csv_paths.items():
        if not csv_path.exists():
            raise FileNotFoundError(
                f"[ERROR] falls_path not exists: {dataset_name}")

        print(f"{dataset_name.title()} dataset file: {csv_path.resolve()}")
        frames.append(pd.read_csv(csv_path, header=None, names=COLUMN_NAMES))

    return pd.concat(frames, ignore_index=True)

def preprocess_dataset(
        df: pd.DataFrame,
        output_dir: pathlib.Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Filter temporary poses, remap labels, and generate diagnostic charts."""

    print(f"Original dataset size: {len(df)} frames")

    # Drop frames that capture the transition between standing and lying (temporary
    # pose, label equal to 0). Final classes used for classification are lying/not lying.
    df_filtered = df[df['label'] != 0].copy()
    print(f"Filtered dataset size (temporary poses removed): {len(df_filtered)} frames")
    print(f"Total sequences : {df['sequence_name'].nunique()}")

    features = df_filtered[FEATURES]
    target = df_filtered['label'].map(LABEL_MAPPING).astype(int)
    groups = df_filtered['sequence_name']

    generate_feature_boxplot(features, output_dir)
    generate_class_balance(target, output_dir)

    return features, target, groups

def make_cross_validator() -> StratifiedGroupKFold:
    """Create the cross-validator used across analysis and training steps"""
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

def train_and_evaluate(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
) -> dict:
    """Train each model with grouped cross-validation and collect metrics and
    predictions."""

    cv = make_cross_validator()
    results = {}

    for model_name, model in MODELS.items():
       fold_metrics = []
       y_true_all, y_pred_all = [], []

       for train_idx, test_idx in tqdm(cv.split(X, y, groups),
                                       total=N_SPLITS,
                                       desc=model_name):
           model.fit(X[train_idx], y[train_idx])
           y_pred = model.predict(X[test_idx])
           y_true = y[test_idx]

           fold_metrics.append({
               'precision' : precision_score(y_true, y_pred),
               'recall'    : recall_score(y_true, y_pred),
               'f1'        : f1_score(y_true, y_pred),
            })

           y_pred_all.append(y_pred)
           y_true_all.append(y_true)

           results[model_name] = {
               'fold_metrics'       : fold_metrics,
               'y_true_all'         : y_true_all,
               'y_pred_all'         : y_pred_all
           }

    return results

def generate_feature_boxplot(features: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Save a boxplot to inspect feature ranges and outliers."""

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(17, 4))
    sns.boxplot(data=features, ax=ax)
    ax.set_title("Feature distribution before training")
    ax.tick_params(axis="x", rotation=5)
    fig.tight_layout()
    fig.savefig(output_dir / "features_distribution_before_training.png", dpi=150)
    plt.close(fig)

def generate_class_balance(target: pd.Series, output_dir: pathlib.Path) -> None:
    """Save a bar plot to evaluate how balanced are target classes."""

    counts = target.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color = ["red", "blue"])
    ax.set_xticks([0, 1], DISPLAY_LABELS)
    ax.set_ylabel("Number of Frames")
    ax.set_title("Target class balance")
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close(fig)

def generate_train_test_distributions(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        output_dir: pathlib.Path,
) -> None:
    """Plot train/test feature distributions for the first CV fold.

    # StratifiedGroupKFold guarantees:
    # - Every sequence appears entirely in either train or test (no leakage).
    # - Class ratio is approximately preserved across folds, given the constraint
    #   of non-overlapping groups between splits.
    """
    cv = make_cross_validator()

    # Generate all fold indices and keep only Fold 1 for this visualization
    folds = list(cv.split(X, y, groups))
    train_idx, test_idx = folds[0]

    X_train, X_test = X[train_idx], X[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Fold 1 - Train vs Test Feature Distributions\n"
        f"train={len(train_idx)} frames / {len(np.unique(groups_train))} sequences\n"
        f"test={len(test_idx)} frames / {len(np.unique(groups_test))} sequences",
        fontsize=13,
        fontweight='bold'
    )

    for feature_ax, ax in enumerate(axes.flatten()):
        ax.hist(X_train[:, feature_ax], bins=20, alpha=0.65, density=True,
                color='blue', label = 'Train')
        ax.hist(X_test[:, feature_ax], bins=20, alpha=0.65, density=True,
                color='red', label = 'Test')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "train_vs_test_feature_distributions.png", dpi=150)
    plt.close(fig)

def print_metrics_table(
        model_name: str,
        metrics: pd.DataFrame) -> None:
    """Print a formatted table of per fold metrics and mean, and standard deviation."""

    print(f"\n {model_name} per fold results:")
    print(f"\n  {'Fold': <6} {'Precision': >10} {'Recall': >8} {'F1': >8}")
    print("  " + "-" * 35)

    for fold_number, row in metrics.iterrows():
        print(
            f"  {fold_number + 1: <6} "
            f"{row['precision']: >10.4f} "
            f"{row['recall']: >8.4f} "
            f"{row['f1']: >8.4f}"
        )

    print("  " + "-" * 35)
    print(
        f"  {'Mean': <6} "
        f"{metrics['precision'].mean(): >10.4f} "
        f"{metrics['recall'].mean(): >8.4f} "
        f"{metrics['f1'].mean(): >8.4f}"
    )
    print(
        f"  {'Std': <6} "
        f"{metrics['precision'].std(): >10.4f} "
        f"{metrics['recall'].std(): >8.4f} "
        f"{metrics['f1'].std(): >8.4f}"
    )

def generate_metric_bars(
        model_name: str,
        metrics: pd.DataFrame,
        output_dir: pathlib.Path,
) -> None:
    """Save a bar chart comparing metrics for each fold."""

    fig, ax = plt.subplots(1, figsize=(16, 5))
    fig.suptitle(f"{model_name} with StratifiedGroupKFold ({N_SPLITS} folds)",
                 fontsize=13, fontweight="bold")

    x = np.arange(N_SPLITS)
    width = 0.2
    for offset, (metric, color) in enumerate(METRIC_STYLES):
        ax.bar(x + offset * width, metrics[metric], width,
               label=metric.upper(), color=color, edgecolor='none', alpha=0.85)
    ax.set_xticks(x + width, [f"Fold {offset + 1}" for offset in range(N_SPLITS)],
                  fontsize=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.legend(fontsize=12, loc="lower right")
    plt.savefig(output_dir / f"{model_name}_barplot_metrics.png", dpi=150)
    plt.close(fig)

def generate_confusion_matrices(
        model_name: str,
        result: dict,
        output_dir: pathlib.Path
) -> None :
    """Save confusion matrix per fold."""

    fig, axes = plt.subplots(1, N_SPLITS, figsize=(16, 5))
    fig.suptitle(f"{model_name}: confusion matrix per fold",
                 fontsize=16, fontweight='bold')

    for fold_index in range(N_SPLITS):
        cm = confusion_matrix(result['y_true_all'][fold_index],
                              result['y_pred_all'][fold_index])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=DISPLAY_LABELS)
        disp.plot(ax=axes[fold_index], colorbar=False, cmap="Blues")
        axes[fold_index].set_title(f"Fold {fold_index + 1}")
        axes[fold_index].set_xlabel("")
        axes[fold_index].set_ylabel("")

    fig.supxlabel("Predicted label", fontsize=13)
    fig.supylabel("True label", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix_per_fold.png", dpi=150)
    plt.close(fig)

def run(args: argparse.Namespace) -> None:
    """Run the full training and evaluation pipeline."""

    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir.resolve()}")

    df = load_and_concat_dataset(data_dir)

    features, target, sequence_name = preprocess_dataset(df, output_dir)
    X = features.values
    y = target.values
    groups = sequence_name.values

    generate_train_test_distributions(X, y, groups, output_dir)
    results = train_and_evaluate(X, y, groups)

    for model_name, result in results.items():
        metrics = pd.DataFrame(result['fold_metrics'])
        print_metrics_table(model_name, metrics)
        generate_metric_bars(model_name, metrics, output_dir)
        generate_confusion_matrices(model_name, result, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Train a posture detector using depth camera features.")
    parser.add_argument("--data_dir",
                        default='../../../share/features_from_depth_camera',
                        help="Path to the dataset containing depth camera features.")
    parser.add_argument("--output_dir",
                        default='output/frame_posture_baselines',
                        help="Path to save the trained posture detector model and data analysis.")
    args = parser.parse_args()

    run(args)
