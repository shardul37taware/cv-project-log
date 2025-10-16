import cv2

cap = cv2.VideoCapture(r"D:\sst\drone survey(1).mp4")

i = 0
while True:
    ret, frame = cap.read()

    
    cv2.imwrite(fr"D:\sst\surveyImages\{i}.jpg", frame)

    i += 1

cap.release()
