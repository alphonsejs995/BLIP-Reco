# save as simple_test.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

print("="*50)
print("ULTRA SIMPLE YOLO TEST")
print("="*50)

# Step 1: Load YOLO (will download fresh)
print("1. Loading YOLOv8...")
model = YOLO('yolov8n.pt')  # This downloads fresh model
print("   ✓ Model loaded!")

# Step 2: Test with simple image
print("\n2. Testing with simple image...")
# Create a test image with you as a "person"
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
# Make a white rectangle (simulating a person)
cv2.rectangle(test_img, (200, 100), (400, 400), (255, 255, 255), -1)

results = model(test_img, conf=0.3, verbose=False)[0]
print(f"   Found {len(results.boxes)} objects in test image")

if len(results.boxes) > 0:
    for i, box in enumerate(results.boxes):
        name = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"   - {i+1}. {name} ({conf:.2f})")
else:
    print("   ⚠ No objects found in test image")

# Step 3: Test with webcam
print("\n3. Testing with webcam (press 'q' to quit)...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("   ✗ No webcam found")
else:
    print("   ✓ Webcam ready - show your face or phone!")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame, conf=0.3, verbose=False)[0]
        
        # Draw boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            label = f"{name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print to console
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"   Detected: {name} ({conf:.2f})")
        
        # Show frame
        cv2.imshow("YOLO Test - Press 'q' to quit", frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*50)
print("TEST COMPLETE")
print("="*50)