import cv2
import depthai as dai
import time
import numpy as np

modelPath = r"griffin/disaster:model-variant-1"  # MobileViT classification blob
confThresh = 0.8

pipeline = dai.Pipeline()

# Camera node
cameraNode = pipeline.create(dai.node.ColorCamera)
cameraNode.setPreviewSize(224, 224)
cameraNode.setInterleaved(False)
cameraNode.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Neural network node
classificationNetwork = pipeline.create(dai.node.NeuralNetwork)
classificationNetwork.setBlobPath(modelPath)
classificationNetwork.input.setBlocking(False)

# Link camera preview to NN input
cameraNode.preview.link(classificationNetwork.input)

# Queues
qRgb = cameraNode.preview.linkOutput()
qClass = classificationNetwork.out.linkOutput()

pipeline.start()

frame = None
predictions = []
startTime = time.monotonic()
counter = 0
color = (255, 255, 255)

# Define your class names
labelMap = ["class0", "class1", "class2"]  # replace with your actual labels

def displayFrame(name, frame):
    if predictions:
        class_id = np.argmax(predictions)
        conf = predictions[class_id]
        if conf >= confThresh:
            text = f"{labelMap[class_id]}: {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

    cv2.imshow(name, frame)

while True:
    inRgb = qRgb.get()
    inClass = qClass.get()

    if inRgb is not None:
        frame = inRgb.getCvFrame()
        cv2.putText(
            frame,
            "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color,
        )

    if inClass is not None:
        # MobileViT outputs a vector of probabilities
        predictions = np.array(inClass.getFirstLayerFp16())
        counter += 1

    if frame is not None:
        displayFrame("rgb", frame)
        print("FPS: {:.2f}".format(counter / (time.monotonic() - startTime)))

    if cv2.waitKey(1) == ord(" "):
        break
