"""Run ONNX structural validation on the exported final model."""

import onnx
import argparse

from pathlib import Path

def run(args: argparse.Namespace) -> None:
    """Load final_model.onnx and validate it with the ONNX checker."""

    model_dir = Path(args.model_dir)

    onnx_model = onnx.load(
        model_dir / "final_model.onnx"
        )

    onnx.checker.check_model(onnx_model)
    print("The ONNX model passed structural validation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Run structural validation on final_model.onnx.")

    parser.add_argument("--onnx_model_dir",
                        default='output/fall_detector_depth/models',
                        help="Directory containing final_model.onnx.")

    args = parser.parse_args()

    run(args)
