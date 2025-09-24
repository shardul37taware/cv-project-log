import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO(r'D:\git\cv-project-log\target\models\ultra_final_final.pt')

# Set confidence threshold
confidence_threshold = 0.8

# Path to your input video file
video_path = r"D:\sst\20250920_133619_1.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object (to save output)
out = cv2.VideoWriter(
    r'D:\git\cv-project-log\target\videos\output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (frame_width, frame_height)
)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video stream.")
        break

    frameResized = cv2.resize(frame, (405, 720))

    # Perform inference
    results = model.predict(source=frameResized, conf=confidence_threshold, verbose=False, imgsz=736)

    # Extract the result from the list
    result = results[0]

    # Plot the results (bounding boxes, labels, confidence)
    annotated_frame = result.plot()

    # Write the frame into output video file
    out.write(annotated_frame)

    # Show the annotated frame (optional)
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("YOLO Inference", 360, 640)  # set custom size
    # cv2.imshow("YOLO Inference", annotated_frame)


    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
