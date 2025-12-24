"""
SMART GLASSES - YOLOv11 ALL CLASSES VERSION
Uses ALL 600+ classes from YOLOv11 without any filtering
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue

print("="*60)
print("SMART GLASSES - YOLOv11 (ALL 600+ CLASSES)")
print("No class filtering - using EVERY object YOLOv11 knows")
print("="*60)

# =================== CONFIGURATION ===================
CONFIG = {
    'model': 'yolo11x.pt',      # YOLOv11 with 600+ classes
    'camera_id': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'confidence': 0.5,           # Balanced threshold
    'detection_interval': 5.0,   # Every 5 seconds to save power
    'show_all_detections': True,
}

# =================== SPEECH MANAGER ===================
class SpeechManager:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.tts_available = True
            print("✓ Speech engine ready")
        except:
            self.tts_available = False
            print("⚠ Speech disabled (install pyttsx3 for audio)")
        
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()
    
    def speak(self, text):
        self.speech_queue.put(text)
    
    def _speech_worker(self):
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                self.is_speaking = True
                
                if self.tts_available:
                    print(f"🔊 {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print(f"[SPEECH]: {text}")
                
                self.is_speaking = False
                self.speech_queue.task_done()
                time.sleep(0.3)
                
            except queue.Empty:
                continue

# =================== MAIN DETECTION SYSTEM ===================
class SmartGlassesSystem:
    def __init__(self, config):
        self.config = config
        self.speech = SpeechManager()
        
        print("\n1. Loading YOLOv11 model...")
        print("   This downloads ~150MB (first time)")
        print("   Please wait...")
        
        self.model = YOLO(config['model'])
        print(f"   ✓ Model loaded: {config['model']}")
        print(f"   ✓ Detecting ALL {len(self.model.names)} object classes")
        
        print("\n2. Initializing camera...")
        self.cap = cv2.VideoCapture(config['camera_id'])
        if not self.cap.isOpened():
            print(f"   ✗ Camera {config['camera_id']} failed")
            self.camera_available = False
            return
        
        self.camera_available = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_height'])
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"   ✓ Camera: {actual_width}x{actual_height}")
        
        self.frame_count = 0
        self.start_time = time.time()
        self.last_detection_time = 0
        self.running = True
        
        print("\n" + "="*60)
        print("SYSTEM READY")
        print(f"Detecting: ALL {len(self.model.names)} YOLOv11 classes")
        print("No class filtering - every object type will be detected")
        print("="*60)
        print("CONTROLS:")
        print("  'q' - Quit")
        print("  's' - Test speech")
        print("  '+' - Increase confidence")
        print("  '-' - Decrease confidence")
        print("  'i' - Show class count info")
        print("="*60 + "\n")
    
    def get_position(self, bbox, frame_width):
        """Determine left/center/right position"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        if center_x < frame_width * 0.4:
            return "left"
        elif center_x > frame_width * 0.6:
            return "right"
        else:
            return "center"
    
    def get_distance(self, bbox_height, frame_height):
        """Estimate distance from object size"""
        if bbox_height > frame_height * 0.6:
            return "very close"
        elif bbox_height > frame_height * 0.3:
            return "close"
        elif bbox_height > frame_height * 0.1:
            return "nearby"
        else:
            return "far"
    
    def run(self):
        if not self.camera_available:
            print("No camera available")
            return
        
        print("Starting detection...")
        print("YOLOv11 is scanning for ALL object types")
        print("-" * 60)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            current_time = time.time()
            display_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate FPS
            fps = self.frame_count / (current_time - self.start_time)
            
            # Run detection at specified interval
            if current_time - self.last_detection_time >= self.config['detection_interval']:
                print(f"\n[Detection Cycle]")
                
                # NO CLASS FILTERING - using ALL YOLOv11 classes
                results = self.model(frame, conf=self.config['confidence'], verbose=False)[0]
                
                self.last_detection_time = current_time
                new_objects_to_speak = []
                unique_classes = set()
                
                # Process ALL detections
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    object_name = self.model.names[class_id]
                    
                    unique_classes.add(object_name)
                    
                    # Calculate position and distance
                    position = self.get_position([x1, y1, x2, y2], frame_width)
                    distance = self.get_distance(y2 - y1, frame_height)
                    
                    # Create detection string
                    detection_string = f"{object_name} on your {position}, {distance}"
                    new_objects_to_speak.append(detection_string)
                    
                    # Draw visualization for ALL detections
                    if self.config['show_all_detections']:
                        # Color code by confidence
                        if confidence > 0.7:
                            color = (0, 255, 0)  # Green - high confidence
                        elif confidence > 0.5:
                            color = (0, 255, 255)  # Yellow - medium
                        else:
                            color = (0, 165, 255)  # Orange - low confidence
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{object_name} {confidence:.2f}"
                        cv2.putText(display_frame, label, (x1, max(20, y1-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Announce objects (limit to avoid overload)
                for obj in new_objects_to_speak[:3]:
                    self.speech.speak(obj)
                
                # Print summary
                print(f"Total detections: {len(results.boxes)}")
                print(f"Unique object types: {len(unique_classes)}")
                if unique_classes:
                    print("Objects found:", ", ".join(list(unique_classes)[:8]))
                    if len(unique_classes) > 8:
                        print(f"  ... and {len(unique_classes)-8} more types")
            
            # Display UI overlay
            time_to_next = self.config['detection_interval'] - (current_time - self.last_detection_time)
            
            # Top-left info panel
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"YOLOv11 - ALL Classes", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Conf: {self.config['confidence']:.2f}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(display_frame, f"Next scan: {max(0, time_to_next):.1f}s", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Position guides
            cv2.line(display_frame, (int(frame_width*0.4), 0),
                    (int(frame_width*0.4), frame_height), (255, 255, 255), 1)
            cv2.line(display_frame, (int(frame_width*0.6), 0),
                    (int(frame_width*0.6), frame_height), (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Smart Glasses - YOLOv11 (No Class Filtering)", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting...")
                self.running = False
            elif key == ord('s'):
                self.speech.speak("System active")
            elif key == ord('+'):
                self.config['confidence'] = min(0.9, self.config['confidence'] + 0.05)
                print(f"Confidence threshold: {self.config['confidence']:.2f}")
            elif key == ord('-'):
                self.config['confidence'] = max(0.1, self.config['confidence'] - 0.05)
                print(f"Confidence threshold: {self.config['confidence']:.2f}")
            elif key == ord('i'):
                print(f"\nYOLOv11 Class Information:")
                print(f"Total classes: {len(self.model.names)}")
                print("Sample classes (first 20):")
                for i in range(min(20, len(self.model.names))):
                    print(f"  {i}: {self.model.names[i]}")
    
    def cleanup(self):
        print("\n" + "="*60)
        print("SYSTEM SUMMARY")
        print("="*60)
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        
        print(f"Total runtime: {total_time:.1f} seconds")
        print(f"Frames processed: {self.frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Model: {self.config['model']}")
        print(f"Classes detected: ALL {len(self.model.names)}")
        print("="*60)
        print("System shutdown complete")

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    print("YOLOv11 SMART GLASSES - ALL CLASSES")
    print("Model will be downloaded automatically")
    print("-" * 60)
    
    system = SmartGlassesSystem(CONFIG)
    
    if system.camera_available:
        try:
            system.run()
        except KeyboardInterrupt:
            print("\nProgram interrupted")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    system.cleanup()