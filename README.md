# Fall Detection

This project develops a fall-detection pipeline using the **UR Fall Detection Dataset**. It begins with frame-level posture classification and extends it with a temporal convolutional neural network (CNN) that analyzes consecutive frames. Using depth-based features provided by the dataset, the temporal model identifies completed fall events rather than relying solely on individual-frame predictions. The final model is exported to ONNX for C++ inference.

The project also includes a computer vision pipeline based on YOLO-Pose, ONNX and OpenCV. This pipeline processes fall-detection images and generate annotated videos. The long-term objective is to extract comparable posture and motion features and use them as inputs to the same temporal convolutional neural network currently trained on the provided depth-based features.

## Project status

- [x] Train and evaluate frame-level posture baselines on existing depth features.
- [x] Trane and evaluate a temporal fall detector on existing depth features.
- [x] Export the temporal detector to ONNX.
- [x] Export and run a YOLO-Pose model with ONNX Runtime.
- [x] Process image sequences and generate annotated videos.
- [ ] Complete features extraction from YOLO-Pose outputs.
- [ ] Train the temporal detector on pose-derived features.
- [ ] Integrate the complete fall-detection pipeline in C++.

## Installation

The Python workflows require PyTorch, Ultralytics, ONNX Runtime, and the scientific Python stack:

```bash
pip install ultralytics onnx onnxscript onnxruntime \ 
    opencv-python numpy tqdm pandas seaborn matplotlib scikit-learn
```

For GPU inference with ONNX Runtime, use `onnxruntime-gpu` instead of the CPU package. Install a PyTorch build compatible with the CUDA version available on your system.

## Dataset

This project uses the **UR Fall Detection Dataset** from the University of Rzeszow:
**Source:** [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)  

**The dataset is not included in this repository.** It is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is intended for non-commercial academic use only. Download it directly from the source.

### Reference
- Bogdan Kwolek, Michal Kepski. *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine, Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607.

### Depth features
The depth-feature workflows expect the following CSV files:
```
<data_dir>/
        <new_folder_dir>
                      urfall-cam0-adls.csv
                      urfall-cam0-falls.csv
```

Each row represents one frame. The original labels are:

- `-1`: not lying;
- `0`: transition;
- `1`: lying.

The eight input features describe body shape and position relative to the floor. Their names are listed in
`src/training/fall_detection_depth/config.py`.

### RGB frames for pose estimation

Download and extract the camera 0 RGB archives. Organize the frames as follows:

```
<data_dir>/
        falls/
            fall-01/
                frame-0.png
                ...
            fall-*/
                frame-0.png
                ...
        adls/
            adl-01/
                frame-0.png
                ...
            adl-*/
                frame-0.png
                ...
```

## Depth-feature workflows

### Frame-level posture

The frame-level script removes transition frames (`label == 0`) and classifies the two stable postures:

- not lying: `-1` remapped to `0`;
- lying:`1`.

It compares Logistic Regression, Random Forest, and Histogram-based Gradient Boosting with grouped cross-validation. Entire sequences remain in one fold to avoid leakage between training and validation.

```bash
python src/training/fall_detection_depth/train_frame_posture_baselines.py \
    --data_dir path/to/data \
    --output_dir path/to/output
```

### Temporal fall detector

The temporal detector uses fixed-size, overlapping windows as input to a one-dimensional CNN.

A fall window is positive when:

1. it belongs to a fall sequence;
2. it contains a transition frame (`0`);
3. a lying frame (`1`) appears later in the same window.

The initial not-lying state (`-1`) does not need to be visible. ADL windows remain negative even when they contain a `0 -> 1` posture transition.

The current temporal configuration uses:

- window size: 55 frames;
- stride: 15 frames;
- three temporal convolutional layers;
- average and maximum temporal pooling.

When the stride does not align with the end of a sequence, one additional fixed-size window is anchored to the final frame. Window sizes therefore remain constant.

#### Evaluation protocol

1. Entire sequences are split into development and held-out test sets.
2. Development sequences are evaluated with `StratifiedGroupKFold`. Windows from one recording cannot appear in both training and validation.
3. The feature scaler is fitted only on the training windows of each fold.
4. Class imbalance is handled with a class weight calculated from each training fold.
5. Out-of-fold probabilities are pooled to select one decision threshold.
6. The threshold must reach a minimum window-level recall. Among valid thresholds, specificity is maximized. 
7. The final number of epochs is the median best epoch across folds.
8. The final model is trained on all development windows and evaluated once on the held-out test set.
9. Threshold-dependent metrics, ranking metrics and confusion matrices are computed for model evaluation.

Train the detector with:

```bash
python src/training/fall_detection_depth/train_temporal_fall_detector.py \
    --data_dir path/to/data \
    --output_dir path/to/output
```

Add `--cuda` to train on CUDA when available.

#### Training outputs

1. `final_model.pth` stores the PyTorch weights, feature-scaler parameters, configuration, decision threshold, and number of training epochs.
2. `TemporalCNN_fold_*.pth` stores the PyTorch weights, feature-scaler parameters, best epoch, best validation loss for each fold.
3. `experiment_results.pth` stores folds summaries, configuration, selected threshold, oof metrics, and held-out test metrics.
4. `final_model.onnx` stores feature scaling, the temporal CNN, and the sigmoid operation. It accepts raw windows with shape `(batch_size, window_size, num_features)` and returns fall probabilities.

The plots include:

- class balance for development and test windows;
- training and validation loss for each fold;
- training loss for the final model;
- OOF probability distributions for positive and negative windows;
- recall and specificity across decision thresholds;
- aggregate OOF and held-out test confusion matrices.

### Validate the exported model

Compare the complete PyTorch and ONNX inference paths on the same raw window:

```bash
python src/training/fall_detection_depth/compare_pytorch_onnx.py \
    --data_dir path/to/data \
    --checkpoint_dir path/to/checkpoint \
    --onnx_model_dir path/to/onnx_model
```

Run the ONNX structural checker:

```bash
python src/training/fall_detection_depth/validate_onnx_model.py \
    --onnx_model_dir path/to/onnx_model
```

## Pose estimation workflow

### Export YOLO-Pose to ONNX

Download a pretrained YOLO-Pose model from
[Ultralytics](https://docs.ultralytics.com/tasks/pose/#models). Then, export it:

```bash
    python src/features_extraction/model_conversion_YOLO_to_ONNX.py \
        --model_path path/to/yolo-pose.pt \
        --output_dir path/to/output
```

### Process image sequences

Run the ONNX pose model over the RGB sequences and generate annotated videos:

```bash
python src/features_extraction/process_dataset_with_YOLO-pose.py \
        --model_path path/to/yolo-pose.onnx \
        --data_dir   path/to/data \
        --output_dir path/to/output \
        --threshold  thr \ # Minimum keypoint confidence to consider a joint valid
```

Add `--cuda` to request the CUDA execution provider when available.
