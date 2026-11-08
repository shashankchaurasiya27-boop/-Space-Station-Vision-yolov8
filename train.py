"""
train.py
========
Cross-platform, reproducible YOLOv8 training script.
Stable on macOS (Apple Silicon MPS), Windows, and Linux.

Author: Shashank Chaurasia
"""

import sys
import argparse
import random
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# -------------------------------------------------
# Reproducibility (safe for Apple Silicon)
# -------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

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
# Dataset sanity check (STRICT)
# -------------------------------------------------
def validate_dataset(data_yaml: Path):
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    if "nc" not in cfg or not isinstance(cfg["nc"], int):
        sys.exit("[ERROR] 'nc' (number of classes) missing or invalid")

    for split in ["train", "val"]:
        if split not in cfg or not cfg[split]:
            sys.exit(f"[ERROR] '{split}' path missing in yolo_params.yaml")

        img_dir = Path(cfg[split]) / "images"
        lbl_dir = Path(cfg[split]) / "labels"

        if not img_dir.exists() or not any(img_dir.glob("*")):
            sys.exit(f"[ERROR] No images found in {img_dir}")

        if not lbl_dir.exists() or not any(lbl_dir.glob("*.txt")):
            sys.exit(f"[ERROR] No labels found in {lbl_dir}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    set_seed(args.seed)

    BASE_DIR = Path(__file__).resolve().parent
    DATA_YAML = BASE_DIR / "yolo_params.yaml"
    RUNS_DIR = BASE_DIR / "runs"

    if not DATA_YAML.exists():
        sys.exit(f"[ERROR] Dataset config not found: {DATA_YAML}")

    validate_dataset(DATA_YAML)

    DEVICE = resolve_device(args.device)

    batch_size = (
        args.batch if args.batch > 0
        else (16 if DEVICE != "cpu" else 8)
    )

    workers = args.workers if DEVICE != "mps" else 0
    amp = False if DEVICE == "mps" else True

    mode = "RESUME" if args.resume else "FRESH"

    print("=" * 76)
    print("YOLOv8 Training – Hack AI Thon")
    print(f"Mode          : {mode}")
    print(f"Device        : {DEVICE}")
    print(f"Model         : {args.weights}")
    print(f"Epochs        : {args.epochs}")
    print(f"Image size    : {args.imgsz}")
    print(f"Batch size    : {batch_size}")
    print(f"Workers       : {workers}")
    print(f"AMP           : {amp}")
    print(f"Optimizer     : {args.optimizer}")
    print(f"LR (0 → f)    : {args.lr0} → {args.lrf}")
    print(f"Mosaic        : {args.mosaic}")
    print("=" * 76)

    # Save experiment config
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = vars(args).copy()
    cfg_dump.update({
        "device": DEVICE,
        "batch": batch_size,
        "workers": workers,
        "amp": amp,
        "mode": mode
    })
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(cfg_dump, f)

    model = YOLO(args.weights)

    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        device=DEVICE,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        mosaic=args.mosaic,
        single_cls=args.single_cls,
        workers=workers,
        seed=args.seed,
        patience=args.patience,
        cache=False,
        amp=amp,
        resume=args.resume,
        project=RUNS_DIR,
        name=args.run_name,
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=1,
    )

    print("\n[SUCCESS] Training completed.")
    print(f"[INFO] Results saved in runs/{args.run_name}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=1e-4)
    parser.add_argument("--mosaic", type=float, default=0.4)
    parser.add_argument("--single_cls", action="store_true")
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="baseline_train")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"]
    )

    args = parser.parse_args()
    main(args)
