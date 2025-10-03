import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

model = dai.NNArchive(r"D:\git\cv-project-log\oak\ultra_final_final.rvc3.tar.xz")

pipeline = dai.Pipeline()
camera = pipeline.create(dai.node.Camera).build()

nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(camera, model)

parser_output_queue = nn_with_parser.out.createOutputQueue()

pipeline.start()

while pipeline.isRunning():
    