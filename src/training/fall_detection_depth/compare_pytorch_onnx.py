"""Compare final-model probabilities produced by PyTorch and ONNX Runtime."""

import argparse
import numpy as np

from pathlib import Path

import onnxruntime as ort

import torch

import onnx_export as oe
import temporal_cnn as tcnn
import temporal_windows as tw

def window_sample_to_tensor(
    sample: tw.WindowSample,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert one window sample into an unscaled batch of size one."""

    raw_window = np.asarray(
        sample.x,
        dtype=np.float32,
    )

    if raw_window.ndim != 2:
        raise ValueError(
            "Expected sample.x to have shape "
            "[window_size, num_features], "
            f"received {raw_window.shape}"
        )

    return torch.from_numpy(raw_window).unsqueeze(0).to(device)

def run(args: argparse.Namespace) -> None:
    """Check PyTorch/ONNX numerical agreement on one temporal window."""

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    onnx_model_dir = Path(args.onnx_model_dir)

    full_df = tw.load_and_concat_dataset(data_dir)

    windows = tw.build_temporal_window_dataset(full_df)
    if not windows:
        raise ValueError("No development windows were created.")

    # Rebuild the complete PyTorch inference path from the final checkpoint.
    checkpoint = torch.load(
        checkpoint_dir / "final_model.pth",
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

    deployment_model = oe.ONNXFallDetector(
        model=base_model,
        scaler_mean=checkpoint["scaler_mean"],
        scaler_scale=checkpoint["scaler_scale"]
    )

    deployment_model.eval()

    sample = windows[30]

    raw_tensor = window_sample_to_tensor(
        sample,
        device="cpu",
    )

    with torch.inference_mode():
        pytorch_probability = deployment_model(raw_tensor)

    # Run the same unscaled window through the exported ONNX graph.
    session = ort.InferenceSession(
        str(onnx_model_dir / "final_model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    raw_window = np.asarray(sample.x, dtype=np.float32)
    raw_batch = raw_window[np.newaxis, :, :]
    onnx_probability = session.run(
        [session.get_outputs()[0].name],
        {
            session.get_inputs()[0].name: raw_batch,
        },
    )[0]

    difference = np.abs(
        pytorch_probability - onnx_probability
    )

    print("PyTorch probability:", pytorch_probability)
    print("ONNX probability:   ", onnx_probability)
    print("Absolute difference:", difference)

    np.testing.assert_allclose(
        pytorch_probability,
        onnx_probability,
        rtol=1e-4,
        atol=1e-5,
    )

    print("PyTorch and ONNX outputs match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Check numerical agreement between the final "
                                     "PyTorch and ONNX models.")

    parser.add_argument("--data_dir",
                        default='../../../share/features_from_depth_camera',
                        help="Directory containing urfall-cam0-falls.csv "
                             "and urfall-cam0-adls.csv.")
    parser.add_argument("--checkpoint_dir",
                        default='output/fall_detector_depth/checkpoints',
                        help="Directory containing final_model.pth.")
    parser.add_argument("--onnx_model_dir",
                        default='output/fall_detector_depth/models',
                        help="Directory containing final_model.onnx.")

    args = parser.parse_args()

    run(args)
