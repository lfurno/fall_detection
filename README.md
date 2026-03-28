# Fall Detection using Pose Estimation

This project explores fall detection using a **YOLO pose** model exported to ONNX. 

This is a work in progress, following the roadmap below

- [x] Run a YOLO pose model and validate it.
- [ ] Features Extraction using YOLO pose output
- [ ] Fall classifier 
- [ ] Evaluation metrics
- [ ] C++ inference


## Installation

```bash
pip install ultralytics onnxruntime opencv-python numpy tqdm
# For GPU inference:
pip install onnxruntime-gpu
```

## Dataset

This project uses the **UR Fall Detection Dataset** from the University of Rzeszow:
**Source:** [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)  

**The dataset is not included in this repository.**  
It is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is intended for non-commercial academic use only.  Download it directly from the link above.

### References
- Bogdan Kwolek, Michal Kepski. *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine, Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607.

### Preparing the data

- Download RGB data from Camera 0 for the sequences you need (e.g. `fall-01-cam0-rgb.zip`, `adl-01-cam0-rgb.zip`).
- Organise the frames in the following layout:

```
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
```

## Model
Download pretrained YOLO pose model from
[Ultralytics](https://docs.ultralytics.com/tasks/pose/#models).

The model needs to be converted to .onnx format. Please, use `model_conversion_YOLO_to_ONNX.py` described below.

## Usage

- `model_conversion_YOLO_to_ONNX.py`: convert a YOLO pose model to ONNX. 
- `process_dataset_with_YOLO_pose.py`: Run the ONNX model over a dataset and produce annotated videos. 

### 1 - Convert the model

Convert a YOLO pose model (.pt) to ONNX format using the Ultralytics export API.

```bash
    python model_conversion_YOLO_to_ONNX.py \
        --model_path path/to/YOLO-model.pt \
        --output_dir path/to/output
```

### 2  - Run inference and produce annotated videos

```bash
python process_dataset_with_YOLO-Pose.py \
        --model_path path/to/model.onnx \
        --data_dir   path/to/data \
        --output_dir path/to/output \
        --threshold  thr \ # Minimum keypoint confidence to render
        [--cuda]  # Enable CUDA execution provider 
```