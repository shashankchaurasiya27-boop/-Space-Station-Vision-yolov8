"""
visualize.py
============
YOLO dataset visualizer for train/val splits.

- OS-agnostic (macOS / Windows / Linux)
- Safe on Apple Silicon
- Judge-friendly & demo-ready
- Robust to missing labels / malformed data

Author: Shashank Chaurasia
"""

import argparse
import sys
import cv2
from pathlib import Path
import yaml

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class YOLOVisualizer:
    MODE_TRAIN = "train"
    MODE_VAL = "val"

    def __init__(self, dataset_root: Path, mode: str = "train"):
        self.dataset_root = dataset_root.resolve()
        self.mode = mode
        self.classes = self._load_classes()
        self._load_split()
        self.index = 0

        print(f"[INFO] Dataset root : {self.dataset_root}")
        print(f"[INFO] Classes      : {len(self.classes)}")
        print(f"[INFO] Split        : {self.mode}")
        print(f"[INFO] Images       : {len(self.images)}")

    # -------------------------------------------------
    # Load class names safely
    # -------------------------------------------------
    def _load_classes(self):
        classes_txt = self.dataset_root / "classes.txt"
        yaml_file = self.dataset_root / "yolo_params.yaml"

        if classes_txt.exists():
            with open(classes_txt, "r") as f:
                return {i: c.strip() for i, c in enumerate(f.readlines())}

        if yaml_file.exists():
            with open(yaml_file, "r") as f:
                cfg = yaml.safe_load(f)
                if "names" in cfg and isinstance(cfg["names"], list):
                    return {i: n for i, n in enumerate(cfg["names"])}

        sys.exit(
            "[ERROR] Class names not found. "
            "Provide classes.txt or names[] in yolo_params.yaml."
        )

    # -------------------------------------------------
    # Load dataset split safely
    # -------------------------------------------------
    def _load_split(self):
        self.images_dir = self.dataset_root / self.mode / "images"
        self.labels_dir = self.dataset_root / self.mode / "labels"

        if not self.images_dir.exists():
            sys.exit(f"[ERROR] Images folder not found: {self.images_dir}")

        self.images = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in VALID_IMAGE_EXTS
        )

        if not self.images:
            sys.exit(f"[ERROR] No valid images found in {self.images_dir}")

        if not self.labels_dir.exists():
            print(f"[WARN] Labels folder not found: {self.labels_dir}")
            print("[WARN] Images will be shown without bounding boxes.")

    # -------------------------------------------------
    # Draw one image safely
    # -------------------------------------------------
    def draw_frame(self):
        img_path = self.images[self.index]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            self.index = (self.index + 1) % len(self.images)
            return None

        label_path = self.labels_dir / img_path.with_suffix(".txt").name

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    try:
                        cls, x, y, w, h = map(float, line.split())
                        h_img, w_img = img.shape[:2]

                        cx, cy = int(x * w_img), int(y * h_img)
                        bw, bh = int(w * w_img), int(h * h_img)

                        x1, y1 = max(cx - bw // 2, 0), max(cy - bh // 2, 0)
                        x2, y2 = min(cx + bw // 2, w_img), min(cy + bh // 2, h_img)

                        cls_id = int(cls)
                        valid = cls_id in self.classes
                        label = self.classes.get(cls_id, "INVALID")

                        color = (0, 255, 0) if valid else (0, 0, 255)

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img,
                            label,
                            (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    except Exception:
                        continue

        overlay = (
            f"{self.mode.upper()} | "
            f"{self.index + 1}/{len(self.images)} | "
            f"{img_path.name}"
        )

        help_text = "A/D: Prev/Next | T/V: Train/Val | Q: Quit"

        cv2.putText(img, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, help_text, (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return cv2.resize(img, (960, 640))

    # -------------------------------------------------
    # Interactive viewer loop
    # -------------------------------------------------
    def run(self):
        print("[INFO] Visualization started")

        while True:
            frame = self.draw_frame()
            if frame is None:
                continue

            cv2.imshow("YOLO Dataset Visualizer", frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                self.index = (self.index + 1) % len(self.images)
            elif key == ord("a"):
                self.index = (self.index - 1) % len(self.images)
            elif key == ord("t"):
                self.mode = self.MODE_TRAIN
                self._load_split()
                self.index = 0
            elif key == ord("v"):
                self.mode = self.MODE_VAL
                self._load_split()
                self.index = 0

        cv2.destroyAllWindows()


# -------------------------------------------------
# CLI Entry Point
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO Dataset Visualizer")
    parser.add_argument(
        "--dataset",
        type=str,
        default=".",
        help="Root directory of YOLO dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Initial split to visualize",
    )

    args = parser.parse_args()

    visualizer = YOLOVisualizer(
        dataset_root=Path(args.dataset),
        mode=args.mode,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
