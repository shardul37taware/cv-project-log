import cv2
import numpy as np
import onnxruntime as ort
import time

# === SETTINGS ===
MODEL_PATH = r"D:\git\cv-project-log\target\models\ultra_final_final.onnx"
CONF_THRESH = 0.4
IMG_SIZE = 640

# === LOAD MODEL ===
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU if available, else CPU
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name
print(f"Running on: {session.get_providers()[0]}")

# === PREPROCESS FUNCTION ===
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB and HWC → CHW
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

# === POSTPROCESS FUNCTION (simplified NMS) ===
def nms(boxes, scores, threshold=0.5):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, threshold)
    return idxs.flatten() if len(idxs) > 0 else []

# === START WEBCAM ===
cap = cv2.VideoCapture(r"D:\sst\drove_vid.mp4")
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    # Inference
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end = time.time()

    # === PARSE OUTPUT ===
    preds = outputs[0][0].T  # shape (num_boxes, 5 or more)
    boxes, scores = [], []

    for det in preds:
        if len(det) < 5:
            continue
        x, y, w, h, conf = det[:5]
        if conf > CONF_THRESH:
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(conf))

    keep = nms(boxes, scores)
    for i in keep:
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{scores[i]:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("YOLO ONNX Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
