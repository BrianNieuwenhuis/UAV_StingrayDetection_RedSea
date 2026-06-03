from ultralytics import YOLO
import os

model = YOLO("PROJECTDIRECTORY/weights/best.pt")

# Directory containing your test images
test_images_dir = 'PROJECTDIRECTORY/Sliced_Raysurveys/RaySurvey_A1_30m_2024-11-18/'
test_images = [os.path.join(test_images_dir, image) for image in os.listdir(test_images_dir)]

# Output directory
save_dir = 'PROJECTDIRECTORY/2024-11-18'

labels_dir = os.path.join(save_dir, 'labels')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)


# Perform predictions
for image_path in test_images:
    results = model.predict(
        image_path,
        imgsz=[1632,2432],
        conf=0.5, #default = 0.25
        max_det=20,
        save=False,  # Prevent automatic saving 
        save_txt=False,  # We'll save manually if detections exist
        save_conf=True,
        show_labels=False,
        show_boxes=True,
        line_width=1
    )

    for result in results:
        # Check if any objects were detected
        if result.boxes is not None and len(result.boxes) > 0:           # put a '#' before this line if you want to save all images
            # Save image with boxes
            result.save(filename=os.path.join(save_dir, os.path.basename(image_path)))
            # Save labels to txt
            # Save labels with same name as image but with .txt extension
            image_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
            label_output_path = os.path.join(labels_dir, image_filename)
            result.save_txt(label_output_path, save_conf=True)
