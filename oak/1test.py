import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera).build()
videoQueue = cam.requestOutput((1920, 1080)).createOutputQueue()


pipeline.start()
while pipeline.isRunning():
    videoIn = videoQueue.get()
    assert isinstance(videoIn, dai.ImgFrame)
    cv2.imshow("full resoluton", videoIn.getCvFrame())

    if cv2.waitKey(1) ==ord(' '):
        break
        
