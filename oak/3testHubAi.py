import cv2
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
import time
import numpy as np

modelPath = r"griffin/target-960:model-variant-1:ec29e9c"
confThresh = 0.8

pipeline = dai.Pipeline()
cameraNode = pipeline.create(dai.node.Camera).build()
detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(cameraNode, dai.NNModelDescription(modelPath))
labelMap = detectionNetwork.getClasses()

qRgb = detectionNetwork.passthrough.createOutputQueue()
qDet = detectionNetwork.out.createOutputQueue()

pipeline.start()

frame = None
detections = []
startTime = time.monotonic()
counter = 0
color = (255, 255, 255)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(name, frame):
    color = (255, 0, 0)
    for detection in detections:
        if detection.confidence >= confThresh:
            bbox = frameNorm(frame,
                            (detection.xmin, detection.ymin, detection.xmax, detection.ymax,)
                            )
            cv2.putText(frame, labelMap[detection.label], 
                        (bbox[0] + 10, bbox[1] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2],bbox[3]), color, 2)

    cv2.imshow(name, frame)

while pipeline.isRunning():
        inRgb: dai.ImgFrame = qRgb.get()
        inDet: dai.ImgDetections = qDet.get()
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

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)
            print("FPS: {:.2f}".format(counter / (time.monotonic() - startTime)))
        if cv2.waitKey(1) == ord(" "):
            pipeline.stop()
            break