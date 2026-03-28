"""
process_dataset_with_YOLO-Pose.py
---------------------------------
Run a YOLO-Pose ONNX model over a dataset of image sequences and write the
annotated output as MP4 videos.
This script uses YOLO-Pose format (17 keypoints).

Expected dataset layout (mirrors the UR Fall Detection Dataset structure 
https://fenix.ur.edu.pl/~mkepski/ds/uf.html after extracting the camera0 
RGB archives):

    <data_dir>/
        falls/
            fall-01/
                frame-0.png
                ...
            fall-num/
                frame-0.png
                ...
        adls/
            adl-01/
                frame-0.png
                ...
            adl-num/
                frame-0.png
                ...

Usage:
    python process_dataset_with_YOLO-Pose.py \
        --model_path path/to/model.onnx \
        --data_dir   path/to/data \
        --output_dir path/to/output \
        --threshold  thr \
        [--cuda]
"""

import argparse
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path

import cv2
import onnxruntime as ort

# Skeleton definition (COCO 17-keypoint layout used by YOLO-Pose)
KEYPOINT_NAMES = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
]

# Pairs of keypoint indices that should be connected by a line.
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head connections
    (0, 5), (0, 6), (5, 6),                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

def unpad_keypoints(
        keypoints_norm: np.ndarray,
        scale: float,
        pad_left: int, 
        pad_top: int,
        orig_h: int,
        orig_w: int
        ) -> list[float]:
    """
    Map normalised [0, 1] keypoint coordinates back to original image pixels.

    Parameters
    ----------
    keypoints_norm: 
        Each row is (x_norm, y_norm, confidence) in [0, 1]. Keypoints are 17.
    scale: 
        Ratio used to resize the original frame to fit inside the square input.
    pad_left, pad_top:
        Pixel offsets added during letterbox padding.
    orig_h, orig_w:
        Height and width of the original (un-resized) frame.

    Returns
    -------
    list of (x_px, y_px, score) tuples in original-frame pixel space.
    """

    result = []
    for x_norm, y_norm, score in keypoints_norm: 
        x_orig = (x_norm - pad_left) / scale
        y_orig = (y_norm - pad_top)  / scale
        x_orig = float(np.clip(x_orig, 0, orig_w - 1))
        y_orig = float(np.clip(y_orig, 0, orig_h - 1))
        result.append((x_orig, y_orig, float(score))) 

    return result

def draw_skeleton(
        frame: np.ndarray,
        keypoints: list[float],
        threshold: float = 0.3
        ) -> None:
    """
    Draw the skeleton (edges and joint dots) onto 'frame' in-place.

    Only joints whose confidence score exceeds 'threshold' are drawn;
    edges are drawn only when both endpoints are confident.

    Parameters
    ----------
    frame:
        image to annotate (modified in-place).
    keypoints:
        17 keypoints in original-frame pixel coordinates.
    threshold:
        minimum confidence score to draw a joint or edge.
    """
    
    for (a, b) in EDGES:
        x1, y1, score1 = keypoints[a]  
        x2, y2, score2 = keypoints[b]
        if score1 > threshold and score2 > threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 0), 2, cv2.LINE_AA)
     
    for _, (x, y, score) in enumerate(keypoints):   
        if score > threshold:
            cv2.circle(frame, (int(x), int(y)), 4,
                       (0, 255, 0), -1, cv2.LINE_AA)   

def run(args):
    """
    Iterate over all image sequences in 'args.data_dir', run pose estimation
    on each frame, and write annotated videos to 'args.output_dir'.

    The expected 'args.data_dir' structure is:
        <data_dir>/<class_name>/<sequence_name>/<frame_files>
    """

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.cuda else ["CPUExecutionProvider"])
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] model_path does not exist: {model_path}")

    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[Pose Estimation] Loaded {model_path.resolve()} "
          f"via {session.get_providers()[0]}")
    
    inp = session.get_inputs()[0]
    print(f"[Pose Estimation] Input shape {inp.shape}")

    label = session.get_outputs()[0]
    print(f"[Pose Estimation] Output shape {label.shape}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] data_dir not exists: {data_dir}")
    print(f"Dataset folder: {data_dir.resolve()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir.resolve()}")

    # Output video frame rate
    fps = 30

    classes = sorted(p for p in data_dir.glob("*") if p.is_dir())    
    for cls in classes:
        sequences = sorted(p for p in cls.glob("*") if p.is_dir())
        
        for seq in sequences:
            files = sorted(seq.glob("*"))
            frames = [f for f in files
                      if f.suffix.lower() in {".jpg", ".jpeg", ".png",
                                              ".bmp", ".tiff"}]
            if not frames:
                warnings.warn(f"Could not find frames in {seq}, skipping.")
                continue
            
            """
            Compute letterbox constants from the first frame
            These are constant for every frame in the sequence, so we only
            compute them once here rather than inside the loop.
            """
            first_frame = cv2.imread(str(frames[0]))
            if first_frame is None:
                raise ValueError(f"Could not read {frames[0]}")
            
            height, width = first_frame.shape[:2]

            # Scale so the longer side fits exactly into the square input.
            squared_frame_size = inp.shape[2]
            scale =  squared_frame_size / max(height, width)
            resized_width, resized_height = int(width * scale), int(height * scale)
            
            # Padding to centre the resized image in the squared frame.
            pad_left = (squared_frame_size - resized_width) // 2
            pad_top  = (squared_frame_size - resized_height) // 2
            pad_bottom = squared_frame_size - resized_height - pad_top
            pad_right  = squared_frame_size - resized_width - pad_left
            
            # Video Wr
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            processed_dir = output_dir / cls.name
            processed_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = str(processed_dir / seq.name) + ".mp4"
            writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height))
            print(f"[Pose Estimation] Saving to: {output_video_path}")
         
            for frame_path in tqdm(frames, total=len(frames), desc=str(seq)):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    warnings.warn(
                        f"Could not read {frame_path}, skipping.")
                    continue
                
                # Preprocess: BGR -> RGB, resize, pad, normalise to [0, 1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb, (resized_width, resized_height))
                frame_padded = cv2.copyMakeBorder(
                    frame_resized,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                tensor = np.expand_dims(
                    frame_padded.astype(np.float32) / 255.0, axis=0)
                tensor = np.transpose(tensor, (0, 3, 1, 2))  # Model is NCHW 
               
                # Inference
                output_model = session.run([label.name], {inp.name: tensor})
                # Shape: (1, num_detections, 5 + 1 + 17 * 3)
                #        [x_c_bbox, y_c_bbox, w_bbox, h_bbox,
                #        conf_value, class_id, kp0...kp16]
                preds = output_model[0][0]
                
                # Keep only person detections (class_id == 0)
                is_person = [row for row in preds if int(row[5]) == 0]

                if not is_person:
                    # No person detected. Write the unmodified frame.
                    writer.write(frame)
                    continue

                # Choose the detection with the highest confidence value.
                best = max(is_person, key=lambda r: float(r[4]))

                # Extract 17 keypoints (x_norm, y_norm, score) and map back
                # to original-frame pixel coordinates.
                keypoints_norm = best[6:].reshape(17, 3).copy()
                keypoints = unpad_keypoints(
                    keypoints_norm, scale, pad_left, pad_top, 
                    height, width)
                              
                draw_skeleton(frame, keypoints, args.threshold)
                writer.write(frame)                
        
            writer.release()
          
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(
        description="Yolo-Pose ONNX inference over image-sequence datasets.")
    parser.add_argument(
        "--model_path",
        default='../share/models/YOLO/yolo26n-pose.onnx',
        help = "Path to the ONNX pose-estimation model.")
    parser.add_argument(
        "--data_dir",
        default='../share/frames',
        help = "Root directory containing class/sequence/frame sub-folders.")
    parser.add_argument(
        "--output_dir",
        default="output",
        help = "Directory where annotated MP4 videos are saved.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum keypoint confidence score to consider valid a joint (default: 0.3).")
    parser.add_argument(
        "--cuda", 
        action="store_true",
        help="Use CUDA execution provider if available")
    args = parser.parse_args()

    run(args)
