import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Path to trained model
MODEL_PATH = "runs/detect/train5/weights/best.pt"  # Change to train5 if needed
# Path to test images
TEST_IMAGES_DIR = "vision/test_images"
# Output file
OUTPUT_FILE = "vision/test_images/results.out"

def get_center_from_box(box):
    # box: [x1, y1, x2, y2]
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return x_center, y_center

def main():
    model = YOLO(MODEL_PATH)
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    results = []
    for img_name in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        result = model(img_path)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
        if len(boxes) == 0:
            results.append(f"{img_name}: Blank")
        else:
            # Use the largest box (by area)
            areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
            largest_idx = int(np.argmax(areas))
            center = get_center_from_box(boxes[largest_idx])
            results.append(f"{img_name}: {center[0]},{center[1]}")
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))
    print(f"Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
