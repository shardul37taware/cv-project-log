
        break

    # Run YOLO inference
    results = model.predict(frame, conf=0.5, verbose=False)

    # Extract detections in [x1, y1, x2, y2, score] format
    detections = []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        detections.append([x1, y1, x2, y2, score])

    # Update SORT tracker
    tracked_objects = tracker.update(np.array(detections))

    # Draw tracks
    for x1, y1, x2, y2, track_id in tracke