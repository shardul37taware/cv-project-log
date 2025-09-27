import cv2
import numpy as np

resize_width = 640

cap = cv2.VideoCapture(r"D:\sst\WhatsApp Video 2025-09-25 at 18.27.06_2358440f.mp4")

ret, frame = cap.read()
# --- Resize frame proportionally ---
orig_h, orig_w = frame.shape[:2]
aspect_ratio = orig_h / orig_w
resize_height = int(resize_width * aspect_ratio)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold 1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold 2", "Parameters", 150, 255, empty)

def getContours(img, imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area >  1000:
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

            print(len(approx))

            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) in [3, 4, 5, 6]:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 5)
                cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(imgContour, f"points: {len(approx)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (resize_width, resize_height))
    imgContour = frame.copy()

    imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold 1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold 2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations = 1)

    getContours(imgDil, imgContour)

    cv2.imshow("Shape Detector", imgContour)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break