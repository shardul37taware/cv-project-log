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
log_df = pd.DataFrame(columns=["id", "x1", "y1", "x2", "y2", "class", "confidence", "time_tracked"])

# --- Video Input ---
cap = cv2.VideoCapture(r"D:\sst\20250920_133619_1_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)  # 🔹 Frames per second

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
print(f"ROI: {roi}, FPS: {fps}")

# --- Track frame counts ---
track_counts = {}

# --- Process Video ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize each frame proportionally
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Run YOLOv8 inference
    results = model.predict(source=frame, show=False, conf=0.8, verbose=False)
    result = results[0]

    detections = np.empty((0, 5))

    # Extract bounding boxes
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    # Update SORT tracker
    TrackResults = tracker.update(detections)

    for tracking in TrackResults:
        x1, y1, x2, y2, track_id = tracking
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)  # ensure integer ID

        # Center point
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Count frames for this ID
        track_counts[track_id] = track_counts.get(track_id, 0) + 1

        # Log only when inside ROI & first time seen
        if roi[0] < cx < roi[2] and roi[1] < cy < roi[3] and track_id not in log_df['id'].values:
            time_tracked = round(track_counts[track_id] / fps, 2) if fps > 0 else 0
            log_df.loc[len(log_df)] = [
                track_id, x1, y1, x2, y2, model.names[cls], round(conf, 2), time_tracked
            ]

        # Draw bounding box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, f'{track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 + SORT Tracking", frame)

    # Exit on ' '
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()

# --- Update time_tracked in final logs ---
for idx, row in log_df.iterrows():
    tid = int(row['id'])
    log_df.at[idx, 'time_tracked'] = track_counts.get(tid, 0) / fps if fps > 0 else 0

# Save logs
log_df.to_csv("0_10_3_02_640.csv", index=False)
