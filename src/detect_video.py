import cv2
from ultralytics import YOLO
import os

def run_detection(video_path, model_path, output_path):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detections_summary = {}

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Run YOLO with tracking enabled (ByteTrack)
        results = model.track(frame, persist=True, verbose=False)

        # Plot boxes and track IDs onto the frame
        annotated_frame = results[0].plot()

        # Log detections
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            
            key = f"{name} (ID: {track_id})" if track_id is not None else name
            detections_summary[key] = max(detections_summary.get(key, 0.0), conf)

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Finished processing {frame_count} frames.")
    print(f"Output saved to: {output_path}")
    print("Detected entities:", detections_summary)

if __name__ == "__main__":
    video_input = os.path.join("data", "tiger_video.mp4")
    model_file = os.path.join("models", "tiger_yolov8.pt")
    video_output = os.path.join("output", "tiger_detected.mp4")
    
    run_detection(video_input, model_file, video_output)
