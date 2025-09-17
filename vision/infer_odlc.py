#!/usr/bin/env python3
"""
Inference script: outputs "x y" pixel coordinates of ODLC center if detected, else "Blank".

Usage:
  python vision/infer_odlc.py --weights runs/detect/train/weights/best.pt --image path/to/input_image.jpg

Note: The input may be any 1920x1080 image file; if your judge passes .in files,
just point --image to that file path.
"""
import argparse
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def select_best_box(result) -> Optional[Tuple[float, float, float, float, float]]:
    """
    From a single image result, return (x0, y0, x1, y1, conf) for the best class-0 box, else None.
    """
    if not hasattr(result, "boxes") or result.boxes is None:
        return None
    boxes = result.boxes
    if boxes.cls is None or boxes.xyxy is None:
        return None

    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros_like(cls, dtype=float)
    xyxy = boxes.xyxy.cpu().numpy()

    mask = (cls == 0)
    if not mask.any():
        return None

    idx = np.argmax(conf[mask])
    chosen = np.where(mask)[0][idx]
    x0, y0, x1, y1 = xyxy[chosen]
    c = float(conf[chosen])
    return float(x0), float(y0), float(x1), float(y1), c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    model = YOLO(args.weights)

    results = model.predict(
        source=args.image,
        conf=args.conf,
        imgsz=960,
        save=True,
        project="runs/predict",
        name="odlc",
        exist_ok=True,
        verbose=False,
        device="auto",
    )

    if not results:
        print("Blank")
        return

    best = select_best_box(results[0])
    if best is None:
        print("Blank")
        return

    x0, y0, x1, y1, _ = best
    cx = int(round((x0 + x1) / 2.0))
    cy = int(round((y0 + y1) / 2.0))
    print(f"{cx} {cy}")

if __name__ == "__main__":
    main()
