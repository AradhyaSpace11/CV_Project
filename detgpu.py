from ultralytics import YOLO
import cv2
import time
import torch

def detect_road_objects(model_path='yolov8n.pt', conf_thresh=0.5, device=0):
    """
    Real-time detection of road-related objects and people using YOLOv8 with GPU acceleration.
    """
    # Check CUDA availability and set device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)  # Use the first CUDA device
    else:
        print("CUDA is not available. Using CPU.")
    
    # Define road-related class IDs
    ROAD_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        9: 'traffic light',
        11: 'stop sign',
        12: 'parking meter'
    }
    
    try:
        # Load YOLOv8 model with CUDA
        print("Loading YOLOv8 model...")
        model = YOLO(model_path)
        
        # Initialize webcam with DirectShow backend for better Windows performance
        print("Opening webcam...")
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        # Set webcam properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Try to get 60fps if camera supports it
        
        # Initialize FPS counter and performance metrics
        fps = 0
        frame_count = 0
        start_time = time.time()
        processing_times = []
        
        print("Starting detection... Press 'q' to quit")
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run YOLOv8 inference with CUDA
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                verbose=False,
                device=0  # Use CUDA device 0
            )
            
            # Get the original frame for custom drawing
            annotated_frame = frame.copy()
            
            # Process detections
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                
                # Only process road-related objects
                if cls_id in ROAD_CLASSES:
                    conf = float(box.conf[0])
                    label = ROAD_CLASSES[cls_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Define colors for different object types
                    if cls_id == 0:  # person
                        color = (0, 255, 0)  # Green
                    elif cls_id in [2, 5, 7]:  # vehicles
                        color = (0, 0, 255)  # Red
                    else:  # other road objects
                        color = (255, 165, 0)  # Orange
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label_text = f"{label} {conf:.2f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Calculate processing time for this frame
            processing_time = time.time() - loop_start
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Add performance metrics to frame
            avg_processing_time = sum(processing_times) / len(processing_times)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Processing Time: {avg_processing_time*1000:.1f}ms", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Road Object Detection (GPU)', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()  # Clear GPU memory
        print("Cleanup complete")

if __name__ == "__main__":
    # Check CUDA version and device information
    if torch.cuda.is_available():
        print(f"CUDA is available. Found device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.backends.cudnn.version()}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Please check your GPU installation.")
    
    # Run the detection
    detect_road_objects()