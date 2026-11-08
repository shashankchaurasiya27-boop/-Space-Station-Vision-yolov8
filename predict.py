"""
predict.py
==========
Cross-platform YOLOv8 inference script.
Optimized for macOS (Apple Silicon MPS), Windows, and Linux.

Author: Shashank Chaurasia
"""

import sys
import argparse
import yaml
import torch
import random
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# -------------------------------------------------
# Reproducibility (lightweight & safe for inference)
# -------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------------------------------
# Device resolution
# -------------------------------------------------
def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    set_seed(args.seed)

    BASE_DIR = Path(__file__).resolve().parent
    RUNS_DIR = BASE_DIR / "runs"
    DATA_YAML = BASE_DIR / "yolo_params.yaml"

    if not DATA_YAML.exists():
        sys.exit(f"[ERROR] yolo_params.yaml not found at {DATA_YAML}")

    # -------------------------------------------------
    # Resolve source images/videos/webcam
    # -------------------------------------------------
    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            sys.exit(f"[ERROR] Specified source path not found: {source_path}")
        source_target = str(source_path)
    else:
        # Load dataset config
        with open(DATA_YAML, "r") as f:
            data_cfg = yaml.safe_load(f)

        if "test" not in data_cfg or not data_cfg["test"]:
            sys.exit("[ERROR] 'test' field missing in yolo_params.yaml and no --source provided")

        TEST_IMAGES_DIR = Path(data_cfg["test"]) / "images"
        if not TEST_IMAGES_DIR.exists() or not any(TEST_IMAGES_DIR.glob("*")):
            # Fallback to assets sample directory if available
            sample_dir = BASE_DIR / "assets" / "sample_predictions"
            if sample_dir.exists() and any(sample_dir.glob("*.jpg")):
                print(f"[WARN] Test folder not found. Falling back to sample assets: {sample_dir}")
                source_target = str(sample_dir)
            else:
                sys.exit(f"[ERROR] Test directory not found ({TEST_IMAGES_DIR}). Please provide --source <path>")
        else:
            source_target = str(TEST_IMAGES_DIR)

    # -------------------------------------------------
    # Resolve model path
    # -------------------------------------------------
    if args.weights:
        model_path = Path(args.weights)
    elif (BASE_DIR / "epoch14_final.pt").exists():
        model_path = BASE_DIR / "epoch14_final.pt"
    elif (RUNS_DIR / "final_train" / "weights" / "best.pt").exists():
        model_path = RUNS_DIR / "final_train" / "weights" / "best.pt"
    else:
        model_path = RUNS_DIR / "baseline_train" / "weights" / "best.pt"

    if not model_path.exists():
        sys.exit(f"[ERROR] Model weights not found: {model_path}")

    DEVICE = resolve_device(args.device)

    print("=" * 68)
    print("YOLOv8 Inference – Space Station Safety Equipment")
    print(f"Device        : {DEVICE}")
    print(f"Source        : {source_target}")
    print(f"Confidence    : {args.conf}")
    print(f"Image Size    : {args.imgsz}")
    print(f"Save TXT      : {args.save_txt}")
    print(f"Model weights : {model_path}")
    print("=" * 68)

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    model = YOLO(model_path)

    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Validate class count
    if "nc" in data_cfg and hasattr(model.model, "nc"):
        if model.model.nc != data_cfg["nc"]:
            sys.exit(
                f"[ERROR] Class mismatch: model has {model.model.nc} classes, "
                f"dataset expects {data_cfg['nc']}"
            )

    # -------------------------------------------------
    # Run inference (AMP explicitly disabled for MPS)
    # -------------------------------------------------
    results = model.predict(
        source=source_target,
        imgsz=args.imgsz,
        conf=args.conf,
        device=DEVICE,
        save=True,
        save_txt=args.save_txt,
        save_conf=True,
        half=False,
        project=RUNS_DIR,
        name=args.run_name,
        exist_ok=True
    )

    total_detections = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            cls_name = model.names.get(cls, str(cls))
            print(f"Detected {cls_name} ({cls}) with confidence {conf:.3f}")
            total_detections += 1

    print(f"\n[INFO] Total objects detected: {total_detections}")
    print("[INFO] Speed metrics:")
    if hasattr(model, "predictor") and model.predictor and hasattr(model.predictor, "speed"):
        print(model.predictor.speed)
    print("\n[SUCCESS] Inference completed.")
    print(f"[INFO] Images + labels saved under: runs/{args.run_name}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Prediction Script")

    parser.add_argument("--source", type=str, default=None, help="Path to image, directory, video, or webcam stream")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="baseline_predict")
    parser.add_argument("--save_txt", action="store_true", help="Save YOLO-format labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"]
    )

    args = parser.parse_args()
    main(args)
