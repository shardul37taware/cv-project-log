import cv2

resize_width = 640

cap = cv2.VideoCapture(r"D:\sst\20250920_133619_1_1.mp4")

ret, frame = cap.read()
# --- Resize frame proportionally ---
orig_h, orig_w = frame.shape[:2]
aspect_ratio = orig_h / orig_w
resize_height = int(resize_width * aspect_ratio)

clahe = cv2.createCLAHE(15.0, (8, 8))

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (resize_width, resize_height))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    frameEq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    frameBlur = cv2.GaussianBlur

    # Show frame
    cv2.imshow("Video", frameEq)

    # Exit on ' '
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

