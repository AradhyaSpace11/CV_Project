# Class IDs for YOLOv8 (COCO dataset):
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck,
# 8: boat, 9: traffic light, 10: fire hydrant, 11: stop sign, 12: parking meter, 13: bench,
# 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 20: elephant, 21: bear,
# 22: zebra, 23: giraffe, 24: backpack, 25: umbrella, 26: handbag, 27: tie, 28: suitcase,
# 29: frisbee, 30: skis, 31: snowboard, 32: sports ball, 33: kite, 34: baseball bat,
# 35: baseball glove, 36: skateboard, 37: surfboard, 38: tennis racket, 39: bottle,
# 40: wine glass, 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl, 46: banana,
# 47: apple, 48: sandwich, 49: orange, 50: broccoli, 51: carrot, 52: hot dog,
# 53: pizza, 54: donut, 55: cake, 56: chair, 57: couch, 58: potted plant, 59: bed,
# 60: dining table, 61: toilet, 62: TV, 63: laptop, 64: mouse, 65: remote,
# 66: keyboard, 67: cell phone, 68: microwave, 69: oven, 70: toaster, 71: sink,
# 72: refrigerator, 73: book, 74: clock, 75: vase, 76: scissors, 77: teddy bear,
# 78: hair drier, 79: toothbrush

from ultralytics import YOLO
import cv2
import time
from collections import deque

def detect_objects(model_path='yolov8n.pt', conf_thresh=0.35, device=0, to_detect=[2]):
    """
    Real-time detection of specified objects based on class IDs in the `to_detect` array.
    """
    total_area_history = deque(maxlen=10)  # Store total area for the last 10 frames
    last_brake_time = 0
    braking_duration = 3
    area_increase_threshold = 1.50  # Adjust this threshold as needed (e.g., 1.5, 2.0)

    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        middle_left = frame_width // 3
        middle_right = 2 * frame_width // 3

        fps = 0
        frame_count = 0
        start_time = time.time()

        print("\nStarting detection... Press 'q' to quit")
        print(f"Area increase threshold: {(area_increase_threshold - 1) * 100:.1f}%")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            current_time = time.time()
            is_braking = (current_time - last_brake_time) < braking_duration

            # Draw vertical lines to define the middle region
            cv2.line(frame, (middle_left, 0), (middle_left, frame_height), (255, 0, 0), 2)
            cv2.line(frame, (middle_right, 0), (middle_right, frame_height), (255, 0, 0), 2)

            # Extract the middle portion of the frame for detection
            middle_frame = frame[:, middle_left:middle_right]

            results = model.predict(source=middle_frame, conf=conf_thresh, verbose=False)
            annotated_frame = frame.copy()

            total_current_area = 0  # Initialize total area for the current frame

            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in to_detect:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Adjust bounding box coordinates to the full frame's context
                    x1 += middle_left
                    x2 += middle_left

                    box_area = (x2 - x1) * (y2 - y1)
                    total_current_area += box_area

                    color = (0, 255, 0)  # Green for detected objects

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    conf = float(box.conf[0])
                    label_text = f"Class {cls_id} {conf:.2f} Area:{box_area / 1000:.1f}k"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            total_area_history.append(total_current_area)  # Add current total area to history

            if len(total_area_history) == 10:  # Check only when we have 10 frames of history
                initial_total_area = total_area_history[0]
                current_total_area = total_area_history[-1]
                if initial_total_area > 0:  # Avoid division by zero
                    area_ratio = current_total_area / initial_total_area
                    if area_ratio > area_increase_threshold and not is_braking:
                        print("\n" + "!" * 50)
                        print(f"BRAKE! Total area increased rapidly! ({area_ratio:.2f}x)")
                        print("!" * 50 + "\n")
                        last_brake_time = current_time
                        is_braking = True

            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if is_braking:
                cv2.putText(annotated_frame, "BRAKING!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time_left = braking_duration - (current_time - last_brake_time)
                cv2.putText(annotated_frame, f"Time left: {time_left:.1f}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Automatic Braking System', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    detect_objects(to_detect=[2, 7])  # Detect cars (class ID 2) and trucks (class ID 7)
