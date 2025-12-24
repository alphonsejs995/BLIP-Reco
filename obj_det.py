"""
MINIMAL YOLOv8 Object Detection Test
Run: python test_detection.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

print("="*50)
print("MINIMAL YOLOv8 OBJECT DETECTION TEST")
print("="*50)

# 1. Load YOLO model
print("1. Loading YOLOv8n model...")
try:
    model = YOLO('yolov8n.pt')  # This downloads if not present
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    print("   Trying to install ultralytics...")
    import os
    os.system("pip install ultralytics")
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print("   ✓ Model loaded after install")

# 2. Initialize webcam
print("\n2. Initializing webcam...")
cap = cv2.VideoCapture(0)  # Change to 1, 2, etc. if needed

if not cap.isOpened():
    print("   ✗ Cannot open camera 0, trying camera 1...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("   ✗ No camera found! Creating test image...")
    USE_TEST_IMAGE = True
else:
    USE_TEST_IMAGE = False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"   ✓ Camera initialized (640x480)")

# 3. Create a test image (if no camera)
if USE_TEST_IMAGE:
    print("\n3. Creating test image with objects...")
    # Create a test image with colored rectangles (simulating objects)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a "person" (white rectangle)
    cv2.rectangle(test_img, (200, 100), (400, 400), (255, 255, 255), -1)
    # Add a "bottle" (blue rectangle)
    cv2.rectangle(test_img, (100, 300), (180, 450), (255, 0, 0), -1)
    # Add a "cell phone" (green rectangle)
    cv2.rectangle(test_img, (450, 200), (550, 300), (0, 255, 0), -1)
    print("   ✓ Test image created with simulated objects")

print("\n" + "="*50)
print("STARTING DETECTION")
print("Controls:")
print("  'q' - Quit")
print("  's' - Save current frame")
print("  't' - Test with static image")
print("="*50)

frame_count = 0
start_time = time.time()
detection_count = 0

while True:
    # Get frame
    if USE_TEST_IMAGE:
        frame = test_img.copy()
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
    
    frame_count += 1
    
    # Calculate FPS
    current_time = time.time()
    fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
    
    # RUN YOLO DETECTION (EVERY FRAME!)
    results = model(frame, conf=0.25, verbose=False)[0]  # LOW confidence threshold
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        
        detections.append({
            'name': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2]
        })
        
        # Draw bounding box (COLOR BASED ON CONFIDENCE)
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)  # Green or Yellow
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Print detection info to console
    if detections:
        detection_count += 1
        print(f"\nFrame {frame_count}: Found {len(detections)} objects")
        for det in detections:
            print(f"  - {det['name']}: {det['confidence']:.2f} at [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}]")
    elif frame_count % 30 == 0:  # Print every 30 frames if no detection
        print(f"Frame {frame_count}: No objects detected")
    
    # Display FPS and detection count
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Total found: {detection_count}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show HELP text
    cv2.putText(frame, "Press 'q' to quit", (10, 400),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow("YOLOv8 Object Detection - MINIMAL TEST", frame)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"detection_test_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"\nSaved screenshot: {filename}")
    elif key == ord('t'):
        print("\n=== TESTING WITH BUILT-IN IMAGE ===")
        test_results = model(test_img, conf=0.25, verbose=False)[0]
        print(f"Test image detections: {len(test_results.boxes)}")
        for box in test_results.boxes:
            print(f"  - {results.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}")

# Cleanup
if not USE_TEST_IMAGE:
    cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("TEST SUMMARY")
print(f"Total frames processed: {frame_count}")
print(f"Total detections made: {detection_count}")
print(f"Average FPS: {fps:.1f}")
print("="*50)