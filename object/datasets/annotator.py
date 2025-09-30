import cv2
import os
import glob

# --- SETTINGS ---
image_folder = r"D:\git\cv-project-log\object\datasets\images"         # folder with your images
output_folder = r"D:\git\cv-project-log\object\datasets\labels"        # folder for YOLO annotations
batch_size = 36                 # number of consecutive images to apply same boxes

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Collect image files (all common formats)
exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_files = []
for e in exts:
    image_files.extend(glob.glob(os.path.join(image_folder, e)))
image_files = sorted(image_files)

if len(image_files) == 0:
    print("❌ No images found in folder:", image_folder)
    exit()

print(f"✅ Found {len(image_files)} images.")

# Global variables
drawing = False
x1, y1, x2, y2 = -1, -1, -1, -1
boxes = []  # list of (x1, y1, x2, y2, class_id)
current_class_id = 0  # default class id
mouse_x, mouse_y = -1, -1  # track mouse position


def mouse_callback(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, boxes, current_class_id, mouse_x, mouse_y
    mouse_x, mouse_y = x, y  # always track mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        boxes.append((x1, y1, x2, y2, current_class_id))


def convert_to_yolo(img_w, img_h, x1, y1, x2, y2, class_id):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = abs(x2 - x1) / img_w
    h  = abs(y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# Loop over batches
for batch_start in range(0, len(image_files), batch_size):
    batch_images = image_files[batch_start:batch_start+batch_size]
    if not batch_images:
        break

    boxes = []  # reset boxes for new batch

    img = cv2.imread(batch_images[0])
    if img is None:
        print(f"⚠️ Skipping unreadable image: {batch_images[0]}")
        continue

    cv2.namedWindow("Draw Bounding Boxes")
    cv2.setMouseCallback("Draw Bounding Boxes", mouse_callback)

    while True:
        display = img.copy()

        # Draw all stored boxes
        for (x1b, y1b, x2b, y2b, cls) in boxes:
            cv2.rectangle(display, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
            cv2.putText(display, f"Class {cls}", (x1b, y1b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw current box (while dragging)
        if drawing:
            cv2.rectangle(display, (x1, y1), (mouse_x, mouse_y), (255, 0, 0), 2)

        # Draw crosshair lines
        if mouse_x >= 0 and mouse_y >= 0:
            cv2.line(display, (mouse_x, 0), (mouse_x, display.shape[0]), (0, 0, 255), 1)
            cv2.line(display, (0, mouse_y), (display.shape[1], mouse_y), (0, 0, 255), 1)

        # Show current class
        cv2.putText(display, f"Current Class: {current_class_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display, f"n: Save and move to next batch.  r: Reset drawn boxes.  q: Quit program.", (10, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Draw Bounding Boxes", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):  # reset boxes
            boxes = []
        elif key == ord("q"):  # quit program
            exit()
        elif key == ord("n"):  # next batch
            break
        elif 48 <= key <= 57:  # number keys 0–9
            current_class_id = key - 48
        elif key == ord("d"):  # delete last box
            if boxes:
                boxes.pop()

    cv2.destroyAllWindows()

    # Apply drawn boxes to all images in batch
    for img_path in batch_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(output_folder,
                                  os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        with open(label_path, "w") as f:
            for (x1b, y1b, x2b, y2b, cls) in boxes:
                yolo_line = convert_to_yolo(w, h, x1b, y1b, x2b, y2b, cls)
                f.write(yolo_line + "\n")

print("✅ All annotations created!")
