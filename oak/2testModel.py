import cv2
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

model = dai.NNArchive(r"D:\git\cv-project-log\oak\ultra_final_final.rvc2_legacy.rvc2.tar.xz")

pipeline = dai.Pipeline()
camera = pipeline.create(dai.node.Camera).build()

nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(camera, model)

frame_queue = nn_with_parser.passthrough.createOutputQueue()
parser_output_queue = nn_with_parser.out.createOutputQueue()

pipeline.start()

while pipeline.isRunning():
    frame_queue_output = frame_queue.get()
    frame = frame_queue_output.getCvFrame()

    cv2.imshow("output", frame)

    if cv2.waitKey(1) == ord(" "):
        break