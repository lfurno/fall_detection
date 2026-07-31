import torch

from torch import nn

import config as cfg


class TemporalCNNFallDetector(nn.Module):
    """Classify fall completion from fixed-length temporal windows.

    Inputs have shape (batch_size, window_size, num_features).
    """

    def __init__(
            self, num_features: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_features,
                      num_features * 2,
                      kernel_size=cfg.KERNEL_SIZE,
                      padding=cfg.PADDING
                      ),
            nn.BatchNorm1d(num_features * 2),
            nn.ReLU(),
            nn.Conv1d(num_features * 2,
                      num_features * 4,
                      kernel_size=cfg.KERNEL_SIZE,
                      padding=cfg.PADDING
                      ),
            nn.BatchNorm1d(num_features*4),
            nn.ReLU(),
            nn.Conv1d(num_features * 4,
                      num_features * 8,
                      kernel_size=cfg.KERNEL_SIZE,
                      padding=cfg.PADDING
                      ),
            nn.BatchNorm1d(num_features * 8),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features * 16,num_features * 4),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(num_features * 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects input of shape
        # (batch_size, num_features, window_size).
        x = x.permute(0, 2, 1)
        x = self.features(x)

        avg_features = torch.mean(x, dim=-1)
        max_features = torch.amax(x, dim=-1)
        x = torch.cat([avg_features, max_features], dim=1)
        logits = self.classifier(x)

        return logits.squeeze(1)
