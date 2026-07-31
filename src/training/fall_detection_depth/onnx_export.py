"""Build and export the deployable ONNX fall detector."""

from pathlib import Path

import torch
from torch import nn

import temporal_cnn as tcnn

class ONNXFallDetector(nn.Module):
    """Scale raw temporal windows and return fall probabilities."""

    def __init__(self, model: nn.Module,
                 scaler_mean: torch.Tensor,
                 scaler_scale: torch.Tensor) -> None:
        super().__init__()

        self.model = model

        # Shape: (1, 1, num_features).
        self.register_buffer(
            "scaler_mean",
            torch.as_tensor(
                scaler_mean,
                dtype=torch.float32
            ).reshape(1, 1, -1)
        )

        self.register_buffer(
            "scaler_scale",
            torch.as_tensor(
                scaler_scale,
                dtype=torch.float32
            ).reshape(1, 1, -1)
        )

    def forward(self, raw_windows: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, window_size, num_features).
        scaler_windows = (
            raw_windows - self.scaler_mean
        ) / self.scaler_scale
        logits = self.model(scaler_windows)
        # Include sigmoid in the exported graph so Python and C++ receive
        # probabilities and can apply the saved decision threshold directly.
        return torch.sigmoid(logits)

def export_final_model_to_onnx(
        checkpoint_path: Path,
        output_path: Path
) -> None:
    """Export the final detector, scaler, and sigmoid to ONNX."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    base_model = tcnn.TemporalCNNFallDetector(
        num_features=len(checkpoint["features_names"])
        )

    base_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    base_model.eval()

    deployment_model = ONNXFallDetector(
        model=base_model,
        scaler_mean=checkpoint["scaler_mean"],
        scaler_scale=checkpoint["scaler_scale"]
    )

    deployment_model.eval()

    example_input = torch.zeros(
        1,
        checkpoint["window_size"],
        len(checkpoint["features_names"])
    )

    with torch.no_grad():
        onnx_program = torch.onnx.export(
            deployment_model,
            (example_input,),
            input_names=["raw_window"],
            output_names=["fall_probability"],
            dynamo=True
        )

    onnx_program.save(output_path)
    print(f"ONNX model saved to: {output_path}")
