# save as smart_glasses_object_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

print("="*50)
print("SMART GLASSES - OPTIMIZED OBJECT DETECTOR")
print("="*50)

# =================== CONFIGURATION ===================
# These are the main knobs you can adjust
CONFIG = {
    'model_path': 'yolov8n.pt',       # Pre-trained model
    'camera_id': 0,                    # Webcam ID (try 1 if 0 fails)
    'frame_width': 640,                # Lower res = faster processing[citation:1][citation:2]
    'frame_height': 480,
    'confidence_threshold': 0.50,      # Higher = fewer, more certain detections[citation:1]
    'detection_interval': 5.0,         # Run detection every N seconds
    'target_classes': [0, 67, 63, 73], # Filter: person, cell phone, laptop, book
}

# =================== INITIALIZATION ===================
print("1. Loading YOLOv8 model...")
model = YOLO(CONFIG['model_path'])
print("   ✓ Model loaded!")

print("\n2. Initializing camera...")
cap = cv2.VideoCapture(CONFIG['camera_id'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])

if not cap.isOpened():
    print(f"   ✗ Cannot open camera {CONFIG['camera_id']}")
    exit()

# Get the actual camera resolution (might differ from requested)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"   ✓ Camera ready: {actual_width}x{actual_height}")

print("\n3. Starting main loop...")
print("   Press 'q' to quit")
print("="*50)

# State variables for timing and control
frame_count = 0
start_time = time.time()
last_detection_time = 0  # Tracks when the last detection cycle ran

try:
    while True:
        # 1. Read a new frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("   ✗ Failed to grab frame.")
            break

        frame_count += 1
        current_time = time.time()

        # 2. Calculate and display simple FPS
        elapsed_total = current_time - start_time
        fps = frame_count / elapsed_total if elapsed_total > 0 else 0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. CORE LOGIC: Run detection only at the specified interval
        time_since_last_detection = current_time - last_detection_time
        if time_since_last_detection >= CONFIG['detection_interval']:
            print(f"\n[Cycle] Running detection... (Waited {time_since_last_detection:.1f}s)")

            # Perform the actual YOLO inference with our settings[citation:1]
            results = model(frame,
                            conf=CONFIG['confidence_threshold'],
                            classes=CONFIG['target_classes'],
                            verbose=False)[0]

            # Reset the timer for the next detection cycle
            last_detection_time = current_time
            detections_this_cycle = []

            # 4. Process results and draw on frame
            for box in results.boxes:
                # Extract box data
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]

                detections_this_cycle.append(f"{class_name} ({confidence:.1f})")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 5. Print results to console for feedback
            if detections_this_cycle:
                print(f"   Found: {', '.join(detections_this_cycle)}")
            else:
                print("   No target objects detected.")

        # 6. Display countdown until next detection
        time_to_next = CONFIG['detection_interval'] - (current_time - last_detection_time)
        countdown_text = f"Next scan in: {max(0, time_to_next):.1f}s"
        cv2.putText(frame, countdown_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 7. Show the final frame with all visualizations
        window_title = f"Smart Glasses Detector | Interval: {CONFIG['detection_interval']}s | Conf: {CONFIG['confidence_threshold']}"
        cv2.imshow(window_title, frame)

        # 8. Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n   Quit signal received.")
            break

except KeyboardInterrupt:
    print("\n   Interrupted by user.")

finally:
    # =================== CLEANUP ===================
    print("\n4. Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    print(f"   Total frames: {frame_count}")
    print(f"   Final FPS: {fps:.1f}")
    print("="*50)
    print("System shutdown.")