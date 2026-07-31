"""Load depth features and build labeled temporal-window datasets."""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import config as cfg


@dataclass
class WindowSample:
    """One fixed-length temporal sample from a single video sequence."""

    sequence_name: str
    start_frame: int
    end_frame: int
    x: np.ndarray  # Shape: (window_size, num_features)
    y: int


class TemporalWindowDataset(Dataset):
    """Expose WindowSample objects as PyTorch tensors."""

    def __init__(self, windows: Sequence[WindowSample]) -> None:
        self.windows = list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.windows[index]
        x = torch.tensor(sample.x, dtype=torch.float32)
        y = torch.tensor(sample.y, dtype=torch.float32)
        return x, y


def load_and_concat_dataset(data_dir: Path) -> pd.DataFrame:
    """Load, validate, and concatenate the fall and ADL CSV files."""

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
        frame = pd.read_csv(csv_path, header=None, names=cfg.COLUMN_NAMES)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    required_columns = ["sequence_name", "frame_number", "label", *cfg.FEATURES]
    missing_values = df[required_columns].isna().sum()
    columns_with_missing_values = missing_values[missing_values > 0]
    if not columns_with_missing_values.empty:
        raise ValueError(
            "[ERROR] Dataset contains missing values in required columns:\n"
            f"{columns_with_missing_values.to_string()}"
        )

    observed_labels = set(df["label"].astype(int))
    unexpected_labels = observed_labels.difference({-1, 0, 1})
    if unexpected_labels:
        raise ValueError(
            f"[ERROR] Unexpected frame labels: {sorted(unexpected_labels)}, "
            "expected only -1, 0, and 1."
        )

    return df


def sequence_type(sequence_name: str) -> str:
    """Infer whether a sequence is a fall or ADL from its filename."""

    normalized_name = str(sequence_name).strip().lower()
    if normalized_name.startswith("fall"):
        return "fall"
    if normalized_name.startswith("adl"):
        return "adl"

    raise ValueError(
        "Cannot infer sequence type from filename "
        f"{sequence_name}. Expected a name beginning "
        "with 'fall' or 'adl'."
    )


def label_window(sequence_name: str, window_df: pd.DataFrame) -> int:
    """Label a window as positive when it contains an ordered fall completion.

    A transition frame (original label: 0) must be followed by a lying frame
    (original label: 1). A transition that does not reach the lying state is
    considered negative.
    """

    if sequence_type(sequence_name) != "fall":
        return 0

    labels = window_df["label"].to_numpy(dtype=np.int64)
    transition_indices = np.flatnonzero(labels == 0)

    for transition_index in transition_indices:
        has_lying_after_transition = np.any(
            labels[transition_index + 1:] == 1)
        if has_lying_after_transition:
            return 1

    return 0


def build_temporal_window_dataset(df: pd.DataFrame) -> list[WindowSample]:
    """Convert each sequence into fixed-size, overlapping temporal windows."""

    windows: list[WindowSample] = []

    for sequence_name, sequence_df in df.groupby('sequence_name'):
        sequence_df = (sequence_df
                       .sort_values("frame_number")
                       .reset_index(drop=True))

        if len(sequence_df) < cfg.WINDOW_SIZE:
            print(
                f"Skipping {sequence_name}: {len(sequence_df)} frames, "
                f"fewer than WINDOW_SIZE={cfg.WINDOW_SIZE}."
            )
            continue

        last_window_start = len(sequence_df) - cfg.WINDOW_SIZE
        window_starts = list(
            range(0, last_window_start + 1, cfg.WINDOW_STRIDE)
        )

        # Add one final fixed-size window when the regular stride does not
        # reach the end of the sequence.
        if window_starts[-1] != last_window_start:
            window_starts.append(last_window_start)

        for window_start in window_starts:
            window_end = window_start + cfg.WINDOW_SIZE
            window_df = sequence_df.iloc[window_start:window_end]

            x = window_df[cfg.FEATURES].to_numpy(dtype=np.float32)
            y = label_window(str(sequence_name), window_df)

            windows.append(
                WindowSample(
                    sequence_name=str(sequence_name),
                    start_frame=int(window_df["frame_number"].iloc[0]),
                    end_frame=int(window_df["frame_number"].iloc[-1]),
                    x=x,
                    y=y
                )
            )

    return windows
