import cv2

resize_width = 640

cap = cv2.VideoCapture(r"D:\sst\20250920_133619_1_1.mp4")

ret, frame = cap.read()
# --- Resize frame proportionally ---
orig_h, orig_w = frame.shape[:2]
aspect_ratio = orig_h / orig_w
resize_height = int(resize_width * aspect_ratio)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (resize_width, resize_height))

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    frameEq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Show frame
    cv2.imshow("YOLOv8 + SORT Tracking", frameEq)

    # Exit on ' '
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

