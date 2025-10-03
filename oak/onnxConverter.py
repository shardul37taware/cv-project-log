from ultralytics import YOLO

# Load your custom model
model = YOLO(r'D:\git\cv-project-log\target\models\ultra_final_final.pt')

# Export to ONNX with fixed parameters
model.export(
    format='onnx',
    dynamic=False,  # IMPORTANT: Disable dynamic axes
    simplify=True,
    opset=12,  # Use opset 12 or higher
    imgsz=640  # Specify fixed input size
)