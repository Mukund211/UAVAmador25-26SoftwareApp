import os
from PIL import Image
import random

# Define directories for images and labels
image_dir = '../data/images/train'  # Folder containing the images
label_dir = '../data/labels/train'  # Folder to save the YOLO labels

# Ensure the label directory exists
os.makedirs(label_dir, exist_ok=True)

# Function to create YOLO label for each image
def create_yolo_label(image_filename):
    image_path = os.path.join(image_dir, image_filename)
    image = Image.open(image_path)
    width, height = image.size
    
    # Randomly decide whether there is an ODLC in the image
    include_odlc = random.choice([True, False])  # Randomly add an ODLC (to simulate real-world cases)
    
    # If ODLC is included, generate random bounding box
    if include_odlc:
        # Generate random coordinates for the bounding box (within image size)
        x_min = random.randint(0, width - 100)
        y_min = random.randint(0, height - 100)
        x_max = x_min + 100  # Fixed width of ODLC
        y_max = y_min + 100  # Fixed height of ODLC

        # Convert coordinates to YOLO format (center x, center y, width, height normalized)
        center_x = ((x_min + x_max) / 2) / width
        center_y = ((y_min + y_max) / 2) / height
        norm_width = (x_max - x_min) / width
        norm_height = (y_max - y_min) / height
        
        label = f"0 {center_x} {center_y} {norm_width} {norm_height}\n"  # Class 0 for ODLC
    else:
        # No ODLC, so label is Blank
        label = "Blank\n"
    
    return label

# Loop through images in the directory and create labels
for image_filename in os.listdir(image_dir):
    if image_filename.endswith('.jpg') or image_filename.endswith('.png'):  # Process only image files
        label = create_yolo_label(image_filename)
        
        # Create label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)
        
        with open(label_path, 'w') as label_file:
            label_file.write(label)

print("Automatic labels created successfully.")
