import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define nodes using v3 API
cam_rgb = pipeline.create(dai.node.Camera)
detection_nn = pipeline.create(dai.node.DetectionNetwork)

# Configure camera using v3 API - different method names
cam_rgb = pipeline.create(dai.node.Camera).build()


# Request output instead of setPreviewSize
camera_output = cam_rgb.requestOutput((640, 640), type=dai.ImgFrame.Type.BGR888p)

# Configure Detection network
detection_nn.setBlobPath(r"D:\git\cv-project-log\oak\ultra_final_final.rvc2_legacy.rvc2.tar.xz")
detection_nn.setConfidenceThreshold(0.5)

# Linking
camera_output.link(detection_nn.input)

# Create output queues directly from nodes (v3 API)
video_queue = camera_output.createOutputQueue(name="video", maxSize=4, blocking=False)
detection_queue = detection_nn.out.createOutputQueue(name="detections", maxSize=4, blocking=False)

# Start pipeline (v3 API)
pipeline.start()

with pipeline:
    while pipeline.isRunning():
        # Get frames and detections
        in_video = video_queue.tryGet()
        in_det = detection_queue.tryGet()

        if in_video is not None:
            # Get frame
            frame = in_video.getCvFrame()
            
            # Process detections if available
            if in_det is not None:
                detections = in_det.detections

                # Draw detections on frame
                for detection in detections:
                    # Get bounding box coordinates
                    x1 = int(detection.xmin * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    x2 = int(detection.xmax * frame.shape[1])
                    y2 = int(detection.ymax * frame.shape[0])

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw label and confidence
                    label_text = f"Class {detection.label} ({detection.confidence:.2f})"
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display output
            cv2.imshow("Detection Network - v3 API", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()