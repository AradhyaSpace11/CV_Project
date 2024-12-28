from ultralytics import YOLO
import cv2
import time
from collections import deque
import numpy as np

def detect_road_objects_enhanced(
    model_path='yolov8n.pt',
    conf_thresh=0.4,
    device=0,
    height_rate_threshold=0.1,
    braking_duration=3,
    danger_zone_percent=0.4
):
    """
    Enhanced real-time detection of road objects with smart braking system.
    Fixed version with proper error handling and ID management.
    """
    ROAD_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign',
        12: 'parking meter'
    }
    
    height_history = {}
    velocity_history = {}
    warning_level = 0
    frame_counter = 0
    fps_history = deque(maxlen=30)
    start_time = time.time()
    last_brake_time = 0
    next_id = 0  # Manual ID counter
    
    try:
        # Initialize model
        print("Loading YOLO model...")
        model = YOLO(model_path)
        print("Model loaded successfully")
        
        # Initialize video capture
        print("Initializing video capture...")
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        # Set camera parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        danger_zone_height = int(frame_height * danger_zone_percent)
        
        middle_left = frame_width // 3
        middle_right = 2 * frame_width // 3

        print("\nEnhanced detection system initialized...")
        print(f"Frame size: {frame_width}x{frame_height}")
        print(f"Danger zone height: {danger_zone_height}px")
        print("Press 'q' to quit, 's' to save snapshot")

        while True:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_counter += 1
            current_time = time.time()
            is_braking = (current_time - last_brake_time) < braking_duration

            # Performance optimization
            if frame_counter % 2 != 0 and not is_braking:
                continue

            # Draw ROI and danger zone
            cv2.line(frame, (middle_left, 0), (middle_left, frame_height), (255, 0, 0), 2)
            cv2.line(frame, (middle_right, 0), (middle_right, frame_height), (255, 0, 0), 2)
            cv2.line(frame, (0, frame_height - danger_zone_height),
                    (frame_width, frame_height - danger_zone_height),
                    (0, 0, 255), 2)

            # Process middle portion of frame
            middle_frame = frame[:, middle_left:middle_right]
            results = model.predict(source=middle_frame, conf=conf_thresh, verbose=False)
            annotated_frame = frame.copy()
            
            if len(results) > 0:  # Check if results exist
                result = results[0]
                current_objects = {}

                # Process detected objects
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    if cls_id in ROAD_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Adjust coordinates to full frame
                        x1 += middle_left
                        x2 += middle_left
                        
                        # Generate ID if not available
                        if hasattr(box, 'id') and box.id is not None:
                            obj_id = int(box.id[0])
                        else:
                            obj_id = next_id
                            next_id += 1
                        
                        height = y2 - y1
                        width = x2 - x1
                        center_y = (y1 + y2) // 2
                        
                        current_objects[obj_id] = {
                            'height': height,
                            'width': width,
                            'center_y': center_y,
                            'cls_id': cls_id
                        }

                        # Calculate danger score
                        danger_score = 0
                        if center_y > (frame_height - danger_zone_height):
                            danger_score += 0.5
                        if obj_id in height_history and len(height_history[obj_id]) > 0:
                            height_change = abs(height - height_history[obj_id][-1])
                            if height_history[obj_id][-1] > 0:  # Prevent division by zero
                                danger_score += min(height_change / height_history[obj_id][-1], 0.5)

                        # Determine color based on danger score
                        color = (
                            int(255 * danger_score),
                            int(255 * (1 - danger_score)),
                            0
                        )

                        # Draw visualization
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        conf = float(box.conf[0])
                        label = ROAD_CLASSES[cls_id]
                        label_text = f"{label} {conf:.2f} Score: {danger_score:.2f}"
                        
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - 20),
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # Trigger braking if necessary
                        if danger_score > 0.7 and not is_braking:
                            print(f"\nBRAKE! High danger score: {danger_score:.2f}")
                            last_brake_time = current_time
                            is_braking = True

                # Update tracking history
                for obj_id, obj_data in current_objects.items():
                    if obj_id not in height_history:
                        height_history[obj_id] = deque(maxlen=5)
                    height_history[obj_id].append(obj_data['height'])

            # Display braking status
            if is_braking:
                cv2.putText(annotated_frame, "BRAKING!", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time_left = braking_duration - (current_time - last_brake_time)
                cv2.putText(annotated_frame, f"Time left: {time_left:.1f}s",
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Calculate and display FPS
            frame_time = time.time() - frame_start_time
            fps_history.append(1 / max(frame_time, 0.001))
            if len(fps_history) > 0:  # Only calculate FPS if we have measurements
                avg_fps = sum(fps_history) / len(fps_history)
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Enhanced Automatic Braking System', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"detection_snapshot_{timestamp}.jpg", annotated_frame)
                print(f"Snapshot saved: detection_snapshot_{timestamp}.jpg")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")
        if len(fps_history) > 0:
            print(f"Average FPS: {sum(fps_history) / len(fps_history):.1f}")
        else:
            print("No FPS data collected")

if __name__ == "__main__":
    detect_road_objects_enhanced()