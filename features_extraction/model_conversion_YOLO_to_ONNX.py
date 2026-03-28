"""
model_conversion_YOLO_to_ONNX.py
--------------------------------
Convert a YOLO-Pose model (.pt) to ONNX format using the Ultralytics export API.

Usage:
    python model_conversion_YOLO_to_ONNX.py \
        --model_path path/to/YOLO-model.pt \
        --output_dir path/to/output
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-Pose .pt model to ONNX format.")
    parser.add_argument("--model_path",
                        default='../share/models/YOLO/yolo26n-pose.pt',
                        help = "Path to the YOLO-Pose .pt model to convert.")
    parser.add_argument("--output_dir",
                        default="../share/models/YOLO/",
                        help = "Directory where the exported ONNX model will be written.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] model_path does not exist: {model_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir.resolve()}")

    model = YOLO(str(model_path))

    model.export(format="onnx")
    print("[Done] Model exported to ONNX.")