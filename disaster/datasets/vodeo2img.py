import cv2

cap = cv2.VideoCapture(r"D:\sst\drove_vid.mp4")

i = 1
while True:
    ret, frame = cap.read()

    if i % 10:
        cv2.imwrite(fr"D:\sst\disaster\dataset raw - Copy\normal\normal ({int(i/10)}).jpg", frame)

    i += 1

cap.release()
