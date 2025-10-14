import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("vtx feed", frame)

    cv2.imshow("vtx feed blur", cv2.GaussianBlur(frame, (3,3), 3))

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()