import cv2
import numpy as np
import os
import glob

# Input folder containing base images
input_dir = r"C:\Users\Admin\OneDrive\Desktop\DATASET\OUTPUT_background\train\REAL_DIRT_TRAIN"
# Output folders
image_dir = r"C:\Users\Admin\OneDrive\Desktop\DATASET\FINAL_TRANING_DATASET\train\images\REAL_DIRT_TRAIN"
label_dir = r"C:\Users\Admin\OneDrive\Desktop\DATASET\FINAL_TRANING_DATASET\train\labels\REAL_DIRT_TRAIN"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Allowed overlap (0 = none, 1 = full overlap allowed)
min_allowed_overlap = 0.2

# style probabilities: solid = 2/3, painted = 1/6, printed = 1/6
style_probs = [2/3, 1/6, 1/6]

def overlap_ratio(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    smaller_area = min(area1, area2) if area1 > 0 and area2 > 0 else 1
    return inter_area / smaller_area

def check_overlap(candidate, boxes, max_ratio):
    for b in boxes:
        if overlap_ratio(candidate, b) > max_ratio:
            return True
    return False

# get list of images
image_paths = glob.glob(os.path.join(input_dir, "."))


for path in image_paths:

    # read base image
    img = cv2.imread(path)
    if img is None:
        continue
    h_img, w_img = img.shape[:2]

    minRatio, maxRatio = 0.03, 0.17
    lb = int(min(h_img, w_img) * minRatio)
    ub = int(min(h_img, w_img) * maxRatio)

    shapes = np.array([3, 4, 5, 6, 20])
    placed_boxes = []
    annotations = []

    for i in range(np.random.randint(2, 7)):
        radius = np.random.randint(lb, ub)
        idx = np.random.randint(0, len(shapes))
        n = shapes[idx]

        for attempt in range(50):
            center = np.random.randint(radius, min(h_img, w_img) - radius, 2).astype(float)
            angle = np.deg2rad(np.random.randint(0, 360))
            colour = tuple(np.random.randint(0, 255, 3).tolist())

            points = []
            for j in range(n):
                theta = angle + j * (2 * np.pi / n)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            candidate_box = (xmin, ymin, xmax, ymax)

            if not check_overlap(candidate_box, placed_boxes, min_allowed_overlap):
                # Decide which style this shape will be
                style = np.random.choice(["solid", "painted", "printed"], p=style_probs)

                if style == "solid":
                    cv2.drawContours(img, [points], 0, colour, -1)

                elif style == "painted":
                    cv2.drawContours(img, [points], 0, colour, 5)

                elif style == "printed":
                    # white base rectangle
                    cv2.rectangle(img, (xmin-10, ymin-10), (xmax+10, ymax+10), (255, 255, 255), -1)
                    # outline on top
                    cv2.drawContours(img, [points], 0, colour, 4)

                placed_boxes.append(candidate_box)

                w = (xmax - xmin) / w_img
                h = (ymax - ymin) / h_img
                xcenter = center[0] / w_img
                ycenter = center[1] / h_img
                
                if n not in [5, 6]:
                    annotations.append(f"{idx} {xcenter:.6f} {ycenter:.6f} {w:.6f} {h:.6f}")
                break

    # save modified image and annotation file with same base name
    if annotations:
        filename = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(image_dir, f"{filename}.jpg"), img)
        with open(os.path.join(label_dir, f"{filename}.txt"), "w") as f:
            f.write("\n".join(annotations))