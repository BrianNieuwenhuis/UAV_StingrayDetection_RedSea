import os
import cv2
import glob
import numpy as np
from sklearn.cluster import DBSCAN
import re
from pathlib import Path

# ---- Parameters ----
input_image_dir = 'C:/INPUTDIRECTORY/valid/images'
input_label_dir = 'C:/INPUTDIRECTORY/valid/labels'
output_image_dir = 'C:/OUTPUTDIRECTORY/valid/images'
output_label_dir = 'C:/OUTPUTDIRECTORY/valid/labels'
crop_w, crop_h = 1344, 1344
padding = 20

# ---- Create output dirs ----
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ---- Helper functions ----
def sanitize_filename(name, max_length=27):
    # Remove illegal characters and truncate long names
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(' ', '_')
    return name[:max_length]

def denormalize_bbox(x_center, y_center, width, height, img_w, img_h):
    """Denormalizes YOLO format (0-1) bounding box coordinates to pixel coordinates."""
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h
    return x_center, y_center, width, height

def normalize_bbox(x_center, y_center, width, height, crop_w, crop_h):
    """Normalizes pixel bounding box coordinates to YOLO format (0-1) relative to a crop."""
    return x_center / crop_w, y_center / crop_h, width / crop_w, height / crop_h

def bbox_fully_within(bbox, x1, y1, x2, y2):
    """Checks if a bounding box is fully contained within a given crop region."""
    bx1, by1, bx2, by2 = bbox
    return bx1 >= x1 and by1 >= y1 and bx2 <= x2 and by2 <= y2

def bbox_intersects(bbox_xyxy, crop_x1, crop_y1, crop_x2, crop_y2):
    """Checks if a bounding box intersects with a given crop region."""
    bx1, by1, bx2, by2 = bbox_xyxy
    # Check for non-intersection: if bounding box is completely to the left, right, above, or below the crop
    if bx1 > crop_x2 or bx2 < crop_x1 or by1 > crop_y2 or by2 < crop_y1:
        return False
    return True

# Helper function to encapsulate clamping and validation logic for a candidate crop
def generate_and_validate_candidate_crop(initial_x1, initial_y1, crop_w, crop_h, img_w, img_h, all_boxes_xyxy):
    """
    Generates a candidate crop from initial coordinates, clamps it to image boundaries,
    and validates it against Rule 3 (no partial bounding boxes).
    Returns (x1, y1, x2, y2, covered_indices) if valid, else None.
    """
    # Calculate crop coordinates based on initial placement and desired dimensions
    # and clamp them within image boundaries.
    
    # Step 1: Calculate the ideal x2, y2 based on initial x1, y1 and crop_w, crop_h.
    # Then clamp these ideal x2, y2 to be within image boundaries.
    crop_x2_ideal = initial_x1 + crop_w
    crop_y2_ideal = initial_y1 + crop_h

    # Ensure crop doesn't extend beyond image right/bottom
    crop_x2 = min(img_w, crop_x2_ideal)
    crop_y2 = min(img_h, crop_y2_ideal)

    # Step 2: Adjust crop_x1, crop_y1 based on the clamped x2, y2 to maintain crop_w, crop_h.
    # This effectively "pushes" the crop left/up if it hit the right/bottom boundary.
    crop_x1 = crop_x2 - crop_w
    crop_y1 = crop_y2 - crop_h
    
    # Step 3: Ensure crop_x1, crop_y1 don't become negative.
    # This handles cases where the image is smaller than the crop, or extreme initial placements.
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)

    # Re-calculate final crop_x2, crop_y2 based on fully clamped crop_x1, crop_y1
    # This ensures the crop dimensions are maintained if possible, or adjusted to image size if img < crop
    crop_x2 = min(img_w, crop_x1 + crop_w)
    crop_y2 = min(img_h, crop_y1 + crop_h)

    # A final adjustment: if the crop dimensions ended up smaller than desired
    # (meaning the image itself was smaller than the crop dimensions),
    # then ensure the crop starts at (0,0) to cover the whole image.
    if (crop_x2 - crop_x1) < crop_w:
        crop_x1 = 0
    if (crop_y2 - crop_y1) < crop_h:
        crop_y1 = 0

    # Identify bounding boxes fully within this candidate crop
    covered_by_this_candidate = set()
    for j in range(len(all_boxes_xyxy)):
        if bbox_fully_within(all_boxes_xyxy[j], crop_x1, crop_y1, crop_x2, crop_y2):
            covered_by_this_candidate.add(j)
    
    # Check for partial bounding box inclusions (Rule 3)
    is_valid_candidate = True
    for j in range(len(all_boxes_xyxy)):
        if bbox_intersects(all_boxes_xyxy[j], crop_x1, crop_y1, crop_x2, crop_y2) and \
           not bbox_fully_within(all_boxes_xyxy[j], crop_x1, crop_y1, crop_x2, crop_y2):
            is_valid_candidate = False
            break
    
    if is_valid_candidate and covered_by_this_candidate:
        # Convert the set of covered indices to a frozenset to make the tuple hashable
        return (crop_x1, crop_y1, crop_x2, crop_y2, frozenset(covered_by_this_candidate))
    return None

# ---- Main processing loop ----
# Get all image files from the input directory
image_files = glob.glob(os.path.join(input_image_dir, '*.jpg')) + \
              glob.glob(os.path.join(input_image_dir, '*.jpeg')) + \
              glob.glob(os.path.join(input_image_dir, '*.png'))

for image_path in image_files:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue

    img_h, img_w, _ = img.shape
    raw_base_name = os.path.basename(image_path).rsplit('.', 1)[0]
    label_path = os.path.join(input_label_dir, raw_base_name + '.txt')
    base_name = sanitize_filename(raw_base_name)  # Only used for saving outputs


    all_boxes = [] # Stores (class_id, xc_norm, yc_norm, w_norm, h_norm) for original YOLO format
    boxes_px = []  # Stores (xc_px, yc_px, w_px, h_px) in pixel coordinates
    boxes_xyxy = []# Stores (x1_px, y1_px, x2_px, y2_px) in pixel coordinates
    labels = []    # Stores class_id for each bounding box

    # Load bounding box annotations if the label file exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                xc_norm, yc_norm, w_norm, h_norm = parts[1:]

                # Denormalize to pixel coordinates
                xc_px, yc_px, w_px, h_px = denormalize_bbox(xc_norm, yc_norm, w_norm, h_norm, img_w, img_h)

                # Convert to xyxy format (top-left x, top-left y, bottom-right x, bottom-right y)
                x1_px = int(xc_px - w_px / 2)
                y1_px = int(yc_px - h_px / 2)
                x2_px = int(xc_px + w_px / 2)
                y2_px = int(yc_px + h_px / 2)

                all_boxes.append((class_id, xc_norm, yc_norm, w_norm, h_norm))
                boxes_px.append((xc_px, yc_px, w_px, h_px))
                boxes_xyxy.append((x1_px, y1_px, x2_px, y2_px))
                labels.append(class_id)
    else:
        print(f"No label file found for {base_name}. Proceeding without annotations.")

    raw_candidate_crops_data = [] # Stores (x1, y1, x2, y2, set_of_bbox_indices_covered_by_this_candidate)

    if not all_boxes:
        # If no bounding boxes, generate a single crop for the entire image
        raw_candidate_crops_data.append((0, 0, img_w, img_h, frozenset())) # Use frozenset for consistency
        
    else:
        # Generate candidate crops from DBSCAN clusters
        centers = np.array([(box[0], box[1]) for box in boxes_px])
        
        if len(centers) > 0:
            dbscan = DBSCAN(eps=max(crop_w, crop_h) / 1, min_samples=1)
            clusters = dbscan.fit_predict(centers)
            num_clusters = max(clusters) + 1
            
            for i in range(num_clusters):
                cluster_bbox_indices = [j for j, cluster_id in enumerate(clusters) if cluster_id == i]
                
                if not cluster_bbox_indices:
                    continue

                # Determine the bounding box that encompasses all boxes in the current cluster
                min_x = min(boxes_xyxy[j][0] for j in cluster_bbox_indices)
                min_y = min(boxes_xyxy[j][1] for j in cluster_bbox_indices)
                max_x = max(boxes_xyxy[j][2] for j in cluster_bbox_indices)
                max_y = max(boxes_xyxy[j][3] for j in cluster_bbox_indices)

                # Calculate the center of the padded bounding box enclosing the cluster
                min_x_padded = max(0, int(min_x - padding))
                min_y_padded = max(0, int(min_y - padding))
                max_x_padded = min(img_w, int(max_x + padding))
                max_y_padded = min(img_h, int(max_y + padding))

                padded_center_x = (min_x_padded + max_x_padded) / 2
                padded_center_y = (min_y_padded + max_y_padded) / 2

                # Calculate initial crop top-left to center the padded cluster
                initial_crop_x1 = int(padded_center_x - crop_w / 2)
                initial_crop_y1 = int(padded_center_y - crop_h / 2)

                candidate = generate_and_validate_candidate_crop(
                    initial_crop_x1, initial_crop_y1, crop_w, crop_h, img_w, img_h, boxes_xyxy
                )
                if candidate:
                    raw_candidate_crops_data.append(candidate)
        
        # --- Generate candidate crops centered around each individual bounding box and its translations ---
        for j in range(len(boxes_xyxy)):
            box_xyxy = boxes_xyxy[j]
            
            # Calculate the center of the padded bounding box for the individual box
            min_x_padded_single = max(0, int(box_xyxy[0] - padding))
            min_y_padded_single = max(0, int(box_xyxy[1] - padding))
            max_x_padded_single = min(img_w, int(box_xyxy[2] + padding))
            max_y_padded_single = min(img_h, int(box_xyxy[3] + padding))

            padded_center_x_single = (min_x_padded_single + max_x_padded_single) / 2
            padded_center_y_single = (min_y_padded_single + max_y_padded_single) / 2

            # Calculate initial crop top-left to center the padded bbox
            initial_crop_x1_centered = int(padded_center_x_single - crop_w / 2)
            initial_crop_y1_centered = int(padded_center_y_single - crop_h / 2)

            # Define the offsets for the additional candidate crops
            offsets = [
                (0, 0),         # Original centered crop
                (-300, -300),   # Translated Left-Top
                (-300, 300),    # Translated Left-Bottom
                (300, -300),    # Translated Right-Top
                (300, 300)      # Translated Right-Bottom
            ]

            for dx, dy in offsets:
                translated_initial_x1 = initial_crop_x1_centered + dx
                translated_initial_y1 = initial_crop_y1_centered + dy

                candidate = generate_and_validate_candidate_crop(
                    translated_initial_x1, translated_initial_y1, crop_w, crop_h, img_w, img_h, boxes_xyxy
                )
                if candidate:
                    raw_candidate_crops_data.append(candidate)


    # --- Filter raw candidates to ensure no partial bounding boxes are included (Rule 3) ---
    # dropping raw candidates that were marked None by 'Rule 3'
    filtered_candidate_crops_data = [
        candidate for candidate in raw_candidate_crops_data if candidate is not None
    ]
    # Remove duplicates by converting to set of tuples, then back to list
    filtered_candidate_crops_data = list(set(filtered_candidate_crops_data))


    # --- Greedy Set Cover Algorithm to minimize crops (Rule 4) ---
    final_crops_coords = [] # Stores (x1, y1, x2, y2) of chosen crops
    covered_bbox_indices = set() # Bounding boxes covered by the *selected* crops
    all_bbox_indices_set = set(range(len(all_boxes)))

    remaining_candidates = list(filtered_candidate_crops_data)

    while len(covered_bbox_indices) < len(all_bbox_indices_set):
        best_candidate = None
        max_newly_covered = -1
        min_overlap_with_chosen = float('inf') # For Rule 5: minimize total appearances

        # Find the candidate crop that covers the most currently uncovered bounding boxes (Rule 4)
        for candidate_info in remaining_candidates:
            cx1, cy1, cx2, cy2, current_candidate_covered_indices = candidate_info
            
            newly_covered = len(current_candidate_covered_indices - covered_bbox_indices)
            
            # Calculate overlap with already chosen crops for tie-breaking (Rule 5)
            overlap_with_chosen = len(current_candidate_covered_indices & covered_bbox_indices)

            if newly_covered > max_newly_covered:
                max_newly_covered = newly_covered
                best_candidate = candidate_info
                min_overlap_with_chosen = overlap_with_chosen 
            elif newly_covered == max_newly_covered: # This is a tie in newly_covered
                # Among candidates with the same max_newly_covered, pick the one with MINIMAL overlap (Rule 5)
                if overlap_with_chosen < min_overlap_with_chosen:
                    min_overlap_with_chosen = overlap_with_chosen
                    best_candidate = candidate_info
        
        if best_candidate is None or max_newly_covered == 0:
            # Fallback for truly uncovered boxes (Rule 1)
            # This happens if a box is isolated and no existing valid candidate can cover it
            # from the generated candidates.
            uncovered_for_fallback = list(all_bbox_indices_set - covered_bbox_indices)
            
            for j in uncovered_for_fallback:
                box_xyxy = boxes_xyxy[j]
                
                # Calculate the center of the padded bounding box for the individual box
                min_x_padded_fallback = max(0, int(box_xyxy[0] - padding))
                min_y_padded_fallback = max(0, int(box_xyxy[1] - padding))
                max_x_padded_fallback = min(img_w, int(box_xyxy[2] + padding))
                max_y_padded_fallback = min(img_h, int(box_xyxy[3] + padding))

                padded_center_x_fallback = (min_x_padded_fallback + max_x_padded_fallback) / 2
                padded_center_y_fallback = (min_y_padded_fallback + max_y_padded_fallback) / 2

                # Calculate initial crop top-left to center the padded bbox
                initial_crop_x1_fallback = int(padded_center_x_fallback - crop_w / 2)
                initial_crop_y1_fallback = int(padded_center_y_fallback - crop_h / 2)

                # Use the helper function for validation
                fallback_candidate = generate_and_validate_candidate_crop(
                    initial_crop_x1_fallback, initial_crop_y1_fallback, crop_w, crop_h, img_w, img_h, boxes_xyxy
                )
                
                if fallback_candidate:
                    # Append only the coordinates from the valid fallback candidate
                    final_crops_coords.append(fallback_candidate[:4])
                    covered_bbox_indices.add(j)
                else:
                    print(f"Warning: Bounding box {j} (original index) could not be cleanly covered by a {crop_w}x{crop_h} crop without partially including another bounding box. This box might not be included in any crop to maintain integrity.")
            break # Exit the while loop as all remaining uncovered boxes are now handled or skipped

        # If a best candidate was found (not None and newly covered > 0), add it to final crops
        if best_candidate and max_newly_covered > 0:
            fcx1, fcy1, fcx2, fcy2, covered_indices_by_best_candidate = best_candidate
            final_crops_coords.append((fcx1, fcy1, fcx2, fcy2))
            
            # Update the set of covered bounding boxes
            covered_bbox_indices.update(covered_indices_by_best_candidate)
            remaining_candidates.remove(best_candidate)
        else: # Should ideally not happen if all_bbox_indices_set is not fully covered
            break


    # --- Generate cropped images and labels from the minimized set of crops ---
    for i, (crop_x1, crop_y1, crop_x2, crop_y2) in enumerate(final_crops_coords):
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        new_labels = []

        # Iterate through ALL original bounding boxes to check for inclusion in the current crop
        # (Rules 1 & 2: If present, annotation must be retained)
        for j in range(len(boxes_xyxy)):
            if bbox_fully_within(boxes_xyxy[j], crop_x1, crop_y1, crop_x2, crop_y2):
                box = all_boxes[j]
                xc, yc, bw, bh = boxes_px[j]
                new_x = xc - crop_x1
                new_y = yc - crop_y1
                norm_x, norm_y, norm_w, norm_h = normalize_bbox(new_x, new_y, bw, bh, crop_w, crop_h)
                new_labels.append(f"{labels[j]} {norm_x} {norm_y} {norm_w} {norm_h}")
                
        # Only save crop if it contains at least one annotation (Rule 1 is implicitly handled here too)
        if len(new_labels) > 0: 
            output_image_path = Path(output_image_dir) / f"{base_name}_crop{i}.jpg"
            output_label_path = Path(output_label_dir) / f"{base_name}_crop{i}.txt"
            cv2.imwrite(str(output_image_path), crop)
            with open(str(output_label_path), 'w') as f:
                for label_line in new_labels:
                    f.write(label_line + '\n')
