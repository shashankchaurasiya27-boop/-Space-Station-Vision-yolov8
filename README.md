# 🚀 Space Station Safety Equipment Vision System (YOLOv8)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-00FFFF?style=flat&logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX%20%7C%20ORT-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![TFLite](https://img.shields.io/badge/Edge-TFLite%20FP16-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Accelerated-Apple%20MPS%20%7C%20CUDA-purple)](#)

> **Autonomous deep learning computer vision system for real-time localization of critical life-support, fire suppression, and emergency equipment inside spacecraft and orbital station modules.**

---

## 🛰️ Project Overview

In pressurized orbital habitats (such as the International Space Station, Lunar Gateway, and future commercial habitats), emergency response time during rapid decompression, toxic leak, thermal runaway, or cabin fire is survival-critical. Under operational stress, astronauts or autonomous intra-vehicular robots (e.g., NASA Astrobee) must rapidly locate and identify safety hardware despite:

* **Extreme Illumination Variations**: From bright solar glare (`vlight`) to nominal illumination (`light`), dimmed modules (`dark`), and complete blackout/emergency backup lighting (`vdark`).
* **High Environmental Clutter**: Floating payload racks, scientific experiments, cargo transfer bags (CTBs), and tethered umbilical cables (`clutter` vs `unclutter`).

This project delivers an end-to-end vision pipeline fine-tuned on custom space station simulation imagery, validated across **1,408 multi-condition test scenarios**, and optimized for embedded, low-SWaP (Size, Weight, and Power) flight computers.

---

## 🎯 Target Safety Classes

The model detects and localizes **7 life-critical equipment classes**:

| Class ID | Class Name | Function in Space Station Operations |
|:---:|:---|:---|
| **0** | `OxygenTank` | Emergency atmospheric supply / primary EVA recharge |
| **1** | `NitrogenTank` | Cabin pressurization and atmospheric buffer regulation |
| **2** | `FirstAidBox` | Medical trauma kit and emergency countermeasures |
| **3** | `FireAlarm` | Smoke, acoustic, and combustion sensor unit |
| **4** | `SafetySwitchPanel` | Electrical isolation, power cutoff, and circuit breaker panel |
| **5** | `EmergencyPhone` | Inter-module hardwired comms & mission control uplink |
| **6** | `FireExtinguisher` | Active fire suppression (e.g., $\text{CO}_2$ / fine water mist) |

---

## 🖼️ Detection Showcase

Visual detections demonstrating robustness across lighting and clutter extremes:

| Scenario: Dimmed + Cluttered Corridor | Scenario: Emergency Dark / Low Illumination |
|:---:|:---:|
| ![Dark Clutter](assets/sample_predictions/000000000_dark_clutter.jpg) | ![Very Dark Clutter](assets/sample_predictions/000000019_vdark_clutter.jpg) |
| *Dark illumination with floating clutter* | *Severe low-light / emergency lighting* |

| Scenario: Nominal Module Illumination | Scenario: High Glare / Direct Light |
|:---:|:---:|
| ![Light Unclutter](assets/sample_predictions/000000002_light_unclutter.jpg) | ![Very Light](assets/sample_predictions/000000004_vlight_unclutter.jpg) |
| *Standard cabin passageway lighting* | *High-exposure direct lighting* |

---

## 📊 Benchmark & Performance Evolution

The model was evaluated by training an initial baseline ($640\times640$) followed by fine-tuning at high resolution ($768\times768$), which proved vital for identifying smaller switches and alarms in long module corridors.

| Metric | Baseline (`baseline_train`) | Final Model (`epoch14_final`) | Final Epoch (Epoch 16) |
|:---|:---:|:---:|:---:|
| **Input Resolution** | $640 \times 640$ | $\mathbf{768 \times 768}$ | $\mathbf{768 \times 768}$ |
| **Precision (B)** | $80.98\%$ | $\mathbf{92.57\%}$ | $\mathbf{95.85\%}$ |
| **Recall (B)** | $58.46\%$ | $\mathbf{75.48\%}$ | $74.65\%$ |
| **mAP @ 0.50** | $65.49\%$ | $\mathbf{83.42\%}$ | $\mathbf{84.20\%}$ |
| **mAP @ 0.50:0.95** | $52.46\%$ | $\mathbf{72.00\%}$ | $\mathbf{72.45\%}$ |
| **Val Box Loss** | $0.6162$ | $\mathbf{0.4479}$ | $0.4506$ |
| **Val Cls Loss** | $0.7867$ | $\mathbf{0.4951}$ | $0.4750$ |

### Model Diagnostics

| Normalized Confusion Matrix | Precision-Recall Curve |
|:---:|:---:|
| ![Confusion Matrix](assets/metrics/confusion_matrix_normalized.png) | ![PR Curve](assets/metrics/BoxPR_curve.png) |

---

## 🧪 Environmental Robustness Matrix

The test suite of **1,408 images** verifies generalizability across systematic operational scenarios:

```
Test Distribution by Lighting:
  dark   (dimmed modules)              : 718 images (51.0%)
  light  (nominal operations)          : 488 images (34.7%)
  vdark  (emergency blackout)          : 120 images (8.5%)
  vlight (direct glare / floodlights)  :  82 images (5.8%)

Test Distribution by Clutter:
  unclutter (open passageways)         : 876 images (62.2%)
  clutter   (operational work areas)   : 532 images (37.8%)
```

---

## 📦 Multi-Platform & Spaceflight Edge Deployment

This repository includes pre-compiled, optimized weights formatted for edge hardware and micro-controllers:

* **PyTorch Weights**: [`epoch14_final.pt`](epoch14_final.pt) ($67.2\text{ MB}$)
* **ONNX Model**: [`epoch14_final.onnx`](epoch14_final.onnx) ($45.0\text{ MB}$) for cross-framework deployment.
* **ONNX Runtime (ORT)**: [`epoch14_final.ort`](epoch14_final.ort) and `epoch14_final.with_runtime_opt.ort` optimized for lightweight runtime execution.
* **Kernel Stripping**: [`epoch14_final.required_operators.config`](epoch14_final.required_operators.config) specifies the exact 15 ONNX operators required (`Concat`, `Sigmoid`, `Softmax`, `Add`, `Div`, `Mul`, `Sub`, `Split`, `Resize`, `Reshape`, `Transpose`, `Conv`, `MaxPool`, `QuickGelu`, `Slice`), enabling custom minimal ORT builds.
* **TensorFlow Lite (TFLite)**:
  * FP32: [`epoch14_final_saved_model/epoch14_final_float32.tflite`](epoch14_final_saved_model/epoch14_final_float32.tflite) ($44.9\text{ MB}$)
  * FP16: [`epoch14_final_saved_model/epoch14_final_float16.tflite`](epoch14_final_saved_model/epoch14_final_float16.tflite) ($22.5\text{ MB}$ — 50% storage and bandwidth savings).
* **Calibration Array**: [`calibration_image_sample_data_20x128x128x3_float32.npy`](calibration_image_sample_data_20x128x128x3_float32.npy) for INT8 Post-Training Quantization (PTQ).

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/shashankchaurasiya27-boop/space-station-vision-yolov8.git
cd space-station-vision-yolov8

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference

Run detection on single images, folders, or sample assets:

```bash
# Run on sample assets
python predict.py --source assets/sample_predictions --conf 0.25 --imgsz 768

# Run on custom image or video
python predict.py --source /path/to/image.jpg --weights epoch14_final.pt --conf 0.35
```

### 3. Interactive Dataset Visualizer

Inspect labeled datasets with bounding box overlays:

```bash
python visualize.py --dataset . --mode train
```
* **Hotkeys**:
  * `A` / `D`: Previous / Next image
  * `T` / `V`: Toggle between Train and Validation splits
  * `Q` / `ESC`: Exit viewer

### 4. Training & Fine-Tuning

Train YOLOv8 with automated Apple Silicon MPS, CUDA, and CPU detection:

```bash
python train.py \
  --weights yolov8s.pt \
  --epochs 25 \
  --imgsz 768 \
  --batch 16 \
  --optimizer AdamW \
  --lr0 1e-4 \
  --mosaic 0.4 \
  --device auto
```

---

## 📂 Repository Structure

```
├── README.md                                               # Documentation
├── LICENSE                                                 # MIT License
├── requirements.txt                                        # Python dependencies
├── classes.txt                                             # 7 safety class names
├── yolo_params.yaml                                        # Dataset configuration
│
├── train.py                                                # Training harness
├── predict.py                                              # Inference & prediction pipeline
├── visualize.py                                            # Dataset visualizer GUI
│
├── epoch14_final.pt                                        # Production PyTorch weights
├── epoch14_final.onnx                                      # ONNX model
├── epoch14_final.ort                                       # ONNX Runtime model
├── epoch14_final.with_runtime_opt.ort                      # Optimized ORT model
├── epoch14_final.required_operators.config                 # Operator pruning configuration
├── calibration_image_sample_data_20x128x128x3_float32.npy # INT8 calibration array
│
├── epoch14_final_saved_model/                              # TFLite & TF SavedModel
│   ├── epoch14_final_float16.tflite                        # FP16 edge model
│   ├── epoch14_final_float32.tflite                        # FP32 edge model
│   └── metadata.yaml                                       # Model parameters
│
├── assets/                                                 # Showcase samples & curves
│   ├── sample_predictions/                                 # Environmental test examples
│   └── metrics/                                            # Performance curves
└── runs/                                                   # Evaluation & training history
```

---

## 👤 Author

* **Shashank Chaurasia** — [GitHub Profile](https://github.com/shashankchaurasiya27-boop)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
