from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np
import pandas as pd

# --- Settings ---
resize_width = 640  # 🔹 Change this to the width you want
xbuffer = 10
ybuffer = 8

# --- Model & Tracker ---
model = YOLO(r"D:\git\cv-project-log\target\models\ultra_final_final.pt")
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.2)

# --- Log DataFrame ---
log_df = pd.DataFrame(columns=["id", "x1", "y1", "x2", "y2", "class", "confidence"])

# --- Video Input ---
cap = cv2.VideoCapture(r"D:\sst\20250920_133619_1_1.mp4")

ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

# --- Resize frame proportionally ---
orig_h, orig_w = frame.shape[:2]
aspect_ratio = orig_h / orig_w
resize_height = int(resize_width * aspect_ratio)

frame = cv2.resize(frame, (resize_width, resize_height))

# --- ROI based on resized frame ---
roi = [xbuffer, ybuffer, resize_width - xbuffer, resize_height - ybuffer]

print(f"Original size: {orig_w}x{orig_h}, Resized: {resize_width}x{resize_height}")
print(f"ROI: {roi}")

# --- Process Video ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize each frame proportionally
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Run YOLOv8 inference
    results = model.predict(source=frame, show=False, conf=0.8)
    result = results[0]

    detections = np.empty((0, 5))

    # Extract bounding boxes
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = box.conf[0]
        cls = int(box.cls[0])

        detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    # Update SORT tracker
    TrackResults = tracker.update(detections)

    for tracking in TrackResults:
        x1, y1, x2, y2, track_id = tracking
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Center point
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Log only when inside ROI & first time seen
        if roi[0] < cx < roi[2] and roi[1] < cy < roi[3] and int(track_id) not in log_df['id'].values:
            log_df.loc[len(log_df)] = [track_id, x1, y1, x2, y2, model.names[cls], float(conf)]

        # Draw bounding box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, f'{int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 + SORT Tracking", frame)

    # Exit on ' '
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()

# Save logs
log_df.to_csv("0_10_3_02_640.csv", index=False)
