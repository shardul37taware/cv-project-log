import cv2
from ultralytics import YOLO
from sort import Sort  # SORT tracker
import numpy as np

# Load YOLOv8 model
model = YOLO(r"D:\git\cv-project-log\target\models\ultra_final_final.pt")

# Initialize SORT tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

# Open video or webcam
cap = cv2.VideoCapture(0)  # use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(frame, conf=0.5, verbose=False)

    # Extract detections in [x1, y1, x2, y2, score] format
    detections = []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        detections.append([x1, y1, x2, y2, score])

    # Ensure proper shape for SORT
    detections = np.array(detections)
    if detections.shape[0] == 0:
        detections = np.empty((0, 5))

    tracked_objects = tracker.update(detections)


    # Draw tracks
    for x1, y1, x2, y2, track_id in tracked_objects:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("YOLO + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
