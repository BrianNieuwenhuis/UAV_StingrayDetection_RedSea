import cv2
import os
import glob

# ---- Settings ----
input_image_dir = "F:/images"  # Directory with original .jpg files
input_label_dir = "F:/labels"  # Directory with YOLO-formatted .txt label files (annotation or inference)
output_image_dir = "F:/visualised" # Directory with visualised images

os.makedirs(output_image_dir, exist_ok=True)

# ---- Class to Color Mapping ----
# Define your class names and their desired hex colors
# You need to know the integer class IDs corresponding to these names from your YOLO dataset.
# Assuming your class IDs are 0, 1, 2, 3, 4, 5 in that order for the listed species.
# Replace with your actual class IDs.

CLASS_COLORS_HEX = {
    # 'Class Name': 'Hex Color'
    'Shark': '#FFFFFF',         # Light Gray
    'Taeniura lymma': '#56B4E9', # Sky Blue
    'Whipray': '#E69F00',       # Light Yellow/Orange
    'Urogymnus granulatus': '#000000', # Black
    'Aetobatus ocellatus': '#000399', # Dark Blue
    'Pastinachus sephen': '#74482F'  # Brown
}

CLASS_ID_TO_NAME = {
    2: 'Shark',
    3: 'Taeniura lymma',
    5: 'Whipray',
    4: 'Urogymnus granulatus',
    0: 'Aetobatus ocellatus',
    1: 'Pastinachus sephen'
}

# IMPORTANT: Map your integer class IDs to the hex colors.
# You need to know the exact class IDs from your dataset's data.yaml or names.txt.
# Adjust these mappings according to your dataset's class IDs.
CLASS_ID_TO_HEX = {
    2: '#FFFFFF',         # Shark
    3: '#56B4E9',         # Taeniura lymma
    5: '#E69F00',         # Whipray
    4: '#000000',         # Urogymnus granulatus
    0: '#000399',         # Aetobatus ocellatus
    1: '#74482F'          # Pastinachus sephen
}

def hex_to_bgr(hex_color):
    """Converts a hex color string to an OpenCV BGR tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0)) # BGR order for OpenCV

# Pre-calculate BGR colors
CLASS_ID_TO_BGR = {
    class_id: hex_to_bgr(hex_code)
    for class_id, hex_code in CLASS_ID_TO_HEX.items()
}

# Define a default color for classes not explicitly listed
DEFAULT_COLOR = (0, 0, 255) # Red in BGR

# ---- Process all image-label pairs ----
for img_path in glob.glob(os.path.join(input_image_dir, '*.jpg')):
    base = os.path.basename(img_path)
    name, _ = os.path.splitext(base)
    label_path = os.path.join(input_label_dir, name + '.txt')

    if not os.path.exists(label_path):
        print(f"Skipping {name}: No label file found.")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {name}: Could not read image.")
        continue

    img_h, img_w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: 
            continue

        try:
            cls_id = int(parts[0])
            # Handle both standard labels (5 cols) and inference labels (6 cols)
            if len(parts) == 6:
                x_center, y_center, w, h, conf = map(float, parts[1:])
                conf_text = f"{conf:.2f}" # Formats 0.8385 to 0.84
            else:
                x_center, y_center, w, h = map(float, parts[1:])
                conf_text = ""
        except ValueError:
            print(f"Skipping malformed data in {label_path}")
            continue

        color = CLASS_ID_TO_BGR.get(cls_id, DEFAULT_COLOR)

        # Denormalize
        x1 = int((x_center - w / 2) * img_w)
        y1 = int((y_center - h / 2) * img_h)
        x2 = int((x_center + w / 2) * img_w)
        y2 = int((y_center + h / 2) * img_h)

        # Draw Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=5)

        # Prepare Label Text (Name + Confidence)
        class_name = CLASS_ID_TO_NAME.get(cls_id, f"ID {cls_id}")
        display_label = f"{class_name} {conf_text}" if conf_text else class_name

        # Font settings for BOLD and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        font_thickness = 5 # Increased for a "Bold" look
        
        # Determine Text Color: If the background (color) is white, use black text.
        # White in BGR is (255, 255, 255)
        text_color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
        
        # Get text size for background box
        (text_w, text_h), baseline = cv2.getTextSize(display_label, font, font_scale, font_thickness)

        # --- Boundary Safety Logic ---
        # 1. Horizontal: Ensure label doesn't go off the right edge
        text_x1 = x1
        if text_x1 + text_w > img_w:
            text_x1 = img_w - text_w - 10 # Shift left if hitting right wall

        # 2. Vertical: If box is too high, move label inside the bounding box
        if y1 - text_h - 20 < 0:
            text_y1 = y1 + text_h + 20 # Move label below the top line
        else:
            text_y1 = y1 # Standard position (above box)

        # Draw Bounding Box (after label so the lines don't overlap the text background)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=8)
        
        # Draw Label Background (Solid rectangle)
        cv2.rectangle(img, (text_x1, text_y1 - text_h - 15), (text_x1 + text_w, text_y1), color, -1)
        
        # Draw Bold Text (White)
        cv2.putText(img, display_label, (text_x1, text_y1 - 10),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        

    # Save the image with annotations
    out_path = os.path.join(output_image_dir, f"{name}_vis.jpg")
    cv2.imwrite(out_path, img)

print("Visualization complete!")
