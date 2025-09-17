#!/usr/bin/env python3
"""
Synthetic ODLC dataset generator.

Generates 1920x1080 (or custom) images with optional ODLC-like targets and YOLOv8 labels.

Usage:
  python vision/generate_synthetic_odlc.py \
    --out vision/datasets/odlc \
    --num-train 800 --num-val 200 \
    --blank-prob 0.3 \
    --img-width 1920 --img-height 1080
"""
import argparse
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import cv2

RNG = np.random.default_rng()

def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def generate_background(width: int, height: int) -> Image.Image:
    noise = RNG.normal(loc=0.5, scale=0.18, size=(height, width, 3)).clip(0, 1)
    noise = (noise * 255).astype(np.uint8)
    ksize = int(max(width, height) * 0.01) | 1
    blurred = cv2.GaussianBlur(noise, (ksize, ksize), sigmaX=0)

    bias = np.array([
        RNG.uniform(0.9, 1.05),
        RNG.uniform(1.0, 1.1),
        RNG.uniform(0.9, 1.0),
    ], dtype=np.float32)
    terrain = np.clip(blurred.astype(np.float32) * bias, 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2.0, height / 2.0
    r = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
    vignette = (1.0 - 0.25 * r).clip(0.75, 1.0)
    terrain = (terrain.astype(np.float32) * vignette[..., None]).clip(0, 255).astype(np.uint8)

    return Image.fromarray(terrain, mode="RGB")

def choose_contrasting_color(bg_patch: np.ndarray) -> Tuple[int, int, int]:
    mean_rgb = bg_patch.mean(axis=(0, 1)) if bg_patch.size else np.array([128, 128, 128], dtype=np.float32)
    target = np.array([RNG.uniform(0, 255) for _ in range(3)], dtype=np.float32)
    direction = np.sign(target - mean_rgb)
    contrasted = np.clip(target + direction * RNG.uniform(50, 100), 0, 255).astype(np.uint8)
    return int(contrasted[0]), int(contrasted[1]), int(contrasted[2])

def draw_shape(
    img: Image.Image,
    bbox_size: Tuple[int, int],
    center: Tuple[int, int],
    outline_color: Tuple[int, int, int],
    fill_color: Tuple[int, int, int],
    shape_kind: str,
) -> Tuple[int, int, int, int]:
    draw = ImageDraw.Draw(img)
    bw, bh = bbox_size
    cx, cy = center
    x0, y0 = int(cx - bw // 2), int(cy - bh // 2)
    x1, y1 = int(cx + bw // 2), int(cy + bh // 2)

    if shape_kind == "rectangle":
        draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=max(1, min(bw, bh) // 20))
        return x0, y0, x1, y1
    elif shape_kind == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=max(1, min(bw, bh) // 20))
        return x0, y0, x1, y1
    elif shape_kind == "triangle":
        p1 = (cx, y0)
        p2 = (x0, y1)
        p3 = (x1, y1)
        draw.polygon([p1, p2, p3], fill=fill_color, outline=outline_color)
        return min(p1[0], x0, x1), min(p1[1], y0, y1), max(p1[0], x0, x1), max(p1[1], y0, y1)
    elif shape_kind == "cross":
        thickness = max(2, int(min(bw, bh) * 0.15))
        draw.rectangle([x0, cy - thickness // 2, x1, cy + thickness // 2], fill=fill_color)
        draw.rectangle([cx - thickness // 2, y0, cx + thickness // 2, y1], fill=fill_color)
        return x0, y0, x1, y1
    else:
        draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color)
        return x0, y0, x1, y1

def clamp_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> Tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)

def yolo_line_from_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int, cls: int = 0) -> str:
    bx = (x0 + x1) / 2.0
    by = (y0 + y1) / 2.0
    bw = (x1 - x0)
    bh = (y1 - y0)
    return f"{cls} {bx / width:.6f} {by / height:.6f} {bw / width:.6f} {bh / height:.6f}"

def generate_one(width: int, height: int, present_prob: float) -> Tuple[Image.Image, List[str]]:
    bg = generate_background(width, height)
    if RNG.random() > present_prob:
        return bg, []

    scale = RNG.uniform(0.04, 0.16)
    min_side = min(width, height)
    obj_w = int(scale * min_side * RNG.uniform(0.9, 1.2))
    obj_h = int(scale * min_side * RNG.uniform(0.9, 1.2))

    margin = int(0.05 * min(width, height))
    cx = RNG.integers(margin + obj_w // 2, width - margin - obj_w // 2)
    cy = RNG.integers(margin + obj_h // 2, height - margin - obj_h // 2)

    np_bg = np.array(bg)
    x0p = max(0, int(cx - obj_w))
    y0p = max(0, int(cy - obj_h))
    x1p = min(width - 1, int(cx + obj_w))
    y1p = min(height - 1, int(cy + obj_h))
    patch = np_bg[y0p:y1p, x0p:x1p]
    fill = choose_contrasting_color(patch)
    outline = tuple(int(v * 0.2) for v in fill)

    shape_kind = random.choice(["rectangle", "circle", "triangle", "cross"])
    x0, y0, x1, y1 = draw_shape(bg, (obj_w, obj_h), (int(cx), int(cy)), outline, fill, shape_kind)
    x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, width, height)

    yolo_line = yolo_line_from_bbox(x0, y0, x1, y1, width, height)
    return bg, [yolo_line]

def save_example(img: Image.Image, labels: List[str], img_path: str, label_path: str) -> None:
    img.save(img_path, format="JPEG", quality=90)
    with open(label_path, "w", encoding="utf-8") as f:
        if labels:
            f.write("\n".join(labels) + "\n")

def write_dataset_yaml(root: str, train_dir: str, val_dir: str) -> None:
    yaml_path = os.path.join(root, "dataset.yaml")
    content = (
        f"path: {os.path.abspath(root)}\n"
        f"train: {os.path.relpath(train_dir, root)}\n"
        f"val: {os.path.relpath(val_dir, root)}\n"
        f"names: ['odlc']\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ODLC dataset")
    parser.add_argument("--out", type=str, default="vision/datasets/odlc", help="Output dataset root directory")
    parser.add_argument("--num-train", type=int, default=1000, help="Number of training images")
    parser.add_argument("--num-val", type=int, default=200, help="Number of validation images")
    parser.add_argument("--blank-prob", type=float, default=0.3, help="Probability an image contains NO object")
    parser.add_argument("--img-width", type=int, default=1920)
    parser.add_argument("--img-height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = args.out
    images_train = os.path.join(root, "images", "train")
    images_val = os.path.join(root, "images", "val")
    labels_train = os.path.join(root, "labels", "train")
    labels_val = os.path.join(root, "labels", "val")

    for d in [images_train, images_val, labels_train, labels_val]:
        ensure_dir(d)

    for i in range(args.num_train):
        img, labels = generate_one(args.img_width, args.img_height, present_prob=1.0 - args.blank_prob)
        img_path = os.path.join(images_train, f"train_{i:05d}.jpg")
        label_path = os.path.join(labels_train, f"train_{i:05d}.txt")
        save_example(img, labels, img_path, label_path)

    for i in range(args.num_val):
        img, labels = generate_one(args.img_width, args.img_height, present_prob=1.0 - args.blank_prob)
        img_path = os.path.join(images_val, f"val_{i:05d}.jpg")
        label_path = os.path.join(labels_val, f"val_{i:05d}.txt")
        save_example(img, labels, img_path, label_path)

    write_dataset_yaml(root, os.path.join(root, "images", "train"), os.path.join(root, "images", "val"))
    print(f"Dataset written to: {os.path.abspath(root)}")

if __name__ == "__main__":
    main()
