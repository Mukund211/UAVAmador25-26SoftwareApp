#!/usr/bin/env python3
"""
Train YOLOv8 on the synthetic ODLC dataset.

Usage:
  python vision/train_yolo.py --data vision/datasets/odlc/dataset.yaml \
    --epochs 50 --imgsz 960 --model yolov8n.pt
"""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model or weights path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=-1, help="-1 auto, else set explicit batch size")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="train",
        exist_ok=True,
        patience=15,
        save=True,
        verbose=True,
        pretrained=True,
        optimizer="auto",
    )

if __name__ == "__main__":
    main()
