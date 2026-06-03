from PIL import Image
import os
import glob
import math

# Parameters
input_dir = "H:/ZenmuseP1_RawData/RaySurvey_A1_30m_2024-05-02/DJI_202405020707_002_RaySurveyIdeal-Charlie"       # Directory with original .jpg files
save_dir = "H:/Sliced_Raysurveys/RaySurvey_A1_30m_2024-05-02"       # Output directory
slice_width, slice_height = 2432, 1632
overlap = 0.2  # 20%

print("Directory exists:", os.path.isdir(input_dir))
print("Contents:", os.listdir(input_dir))

# Stride based on overlap
stride_x = math.ceil(slice_width * (1 - overlap))  # = 1638
stride_y = math.ceil(slice_height * (1 - overlap)) # = 1092

# Make sure output directory exists
os.makedirs(save_dir, exist_ok=True)

# Get all jpg images in input_dir
image_paths = [
    os.path.join(input_dir, fname)
    for fname in os.listdir(input_dir)
    if fname.lower().endswith(".jpg")
]


# Process each image
for image_path in image_paths:
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # filename without extension
    img_w, img_h = img.size
    
    slice_count = 0
    # Determine the number of rows and columns of slices
    num_cols = math.ceil( (img_w - slice_width + stride_x) / stride_x ) if img_w >= slice_width else 1
    num_rows = math.ceil( (img_h - slice_height + stride_y) / stride_y ) if img_h >= slice_height else 1
    # Ensure at least one slice if image is smaller than slice_width/height
    if img_w < slice_width: num_cols = 1
    if img_h < slice_height: num_rows = 1
    
    count = 1
    for row in range(num_rows):
        for col in range(num_cols):
            x = int(col * stride_x)
            y = int(row * stride_y)
            box = (x, y, x + slice_width, y + slice_height)

            # Crop the slice
            slice_img = img.crop(box)

            # Save with filename_base_1.jpg to filename_base_25.jpg
            save_path = os.path.join(save_dir, f"{base_name}_{count}.jpg")
            slice_img.save(save_path, quality=100)
            count += 1

    print(f"Processed: {base_name}")

print("All images processed.")
