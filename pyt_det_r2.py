"""
SMART GLASSES - COMPLETE INDOOR OBJECT DETECTOR
Includes ALL indoor-relevant YOLOv8 classes
Runs detection every 5 seconds to save power
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue

print("="*60)
print("SMART GLASSES - INDOOR OBJECT DETECTION SYSTEM")
print("="*60)

# =================== COMPREHENSIVE INDOOR CLASS CONFIG ===================
CONFIG = {
    'model': 'yolov8n.pt',
    'camera_id': 0,
    'frame_width': 640,
    'frame_height': 480,
    'confidence': 0.45,
    'detection_interval': 5.0,  # Every 5 seconds
    
    # ✅ ALL INDOOR-RELEVANT CLASSES (39 out of 80 total)
    'indoor_classes': [
        # People & Living
        0,    # person
        1,    # bicycle (could be indoors)
        
        # Furniture
        56,   # chair
        57,   # couch/sofa
        58,   # potted plant
        59,   # bed
        60,   # dining table
        61,   # toilet
        62,   # tv
        63,   # laptop
        64,   # mouse
        65,   # remote
        66,   # keyboard
        67,   # cell phone
        68,   # microwave
        69,   # oven
        70,   # toaster
        71,   # sink
        72,   # refrigerator
        
        # Household Items
        39,   # bottle
        40,   # wine glass
        41,   # cup
        42,   # fork
        43,   # knife
        44,   # spoon
        45,   # bowl
        46,   # banana
        47,   # apple
        48,   # sandwich
        49,   # orange
        50,   # broccoli
        51,   # carrot
        52,   # hot dog
        53,   # pizza
        54,   # donut
        55,   # cake
        
        # Stationery & Media
        73,   # book
        74,   # clock
        75,   # vase
        76,   # scissors
        77,   # teddy bear
        78,   # hair drier
        79,   # toothbrush
        
        # Additional useful items
        13,   # bench (could be indoor)
        14,   # bird (pet)
        15,   # cat (pet)
        16,   # dog (pet)
        17,   # horse (unlikely but included)
        18,   # sheep (unlikely)
        19,   # cow (unlikely)
        20,   # elephant (toy)
        21,   # bear (toy)
        22,   # zebra (toy)
        23,   # giraffe (toy)
    ],
    
    # Class names for display
    'class_names': {
        0: "person", 1: "bicycle", 13: "bench", 14: "bird", 15: "cat",
        16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
        21: "bear", 22: "zebra", 23: "giraffe", 39: "bottle", 40: "wine glass",
        41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
        51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
        56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
        61: "toilet", 62: "TV", 63: "laptop", 64: "mouse", 65: "remote",
        66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
        71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
        76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
    }
}

# =================== SPEECH MANAGER ===================
class SpeechManager:
    """Handles text-to-speech output"""
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.tts_available = True
            print("✓ Speech engine ready")
        except:
            self.tts_available = False
            print("⚠ Speech disabled (pyttsx3 not available)")
        
        # Start speech thread
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()
    
    def speak(self, text):
        """Add text to speech queue"""
        self.speech_queue.put(text)
    
    def _speech_worker(self):
        """Process speech in background thread"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                self.is_speaking = True
                
                if self.tts_available:
                    print(f"🔊 Speaking: {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print(f"[SPEECH]: {text}")
                
                self.is_speaking = False
                self.speech_queue.task_done()
                time.sleep(0.5)  # Pause between speeches
                
            except queue.Empty:
                continue

# =================== MAIN DETECTION SYSTEM ===================
class SmartGlassesSystem:
    def __init__(self, config):
        self.config = config
        self.speech = SpeechManager()
        
        print("\n" + "="*60)
        print("INITIALIZING SYSTEM...")
        print("="*60)
        
        # Load YOLO model
        print("1. Loading YOLOv8 model...")
        self.model = YOLO(config['model'])
        print(f"   ✓ Model loaded")
        print(f"   Detecting {len(config['indoor_classes'])} indoor object types")
        
        # Initialize camera
        print("\n2. Initializing camera...")
        self.cap = cv2.VideoCapture(config['camera_id'])
        if not self.cap.isOpened():
            print(f"   ✗ Camera {config['camera_id']} failed, trying camera 1...")
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            print("   ✗ No camera available")
            self.camera_available = False
            return
        
        self.camera_available = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_height'])
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"   ✓ Camera ready: {actual_width}x{actual_height}")
        
        # State tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_detection_time = 0
        self.running = True
        
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  'q' - Quit program")
        print("  's' - Test speech")
        print("  '+' - Increase confidence (current: {:.2f})".format(config['confidence']))
        print("  '-' - Decrease confidence")
        print("  'd' - Toggle debug display")
        print("  'i' - Show indoor class list")
        print("="*60)
    
    def get_position_text(self, bbox, frame_width):
        """Convert bounding box position to left/center/right"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        if center_x < frame_width * 0.4:
            return "left"
        elif center_x > frame_width * 0.6:
            return "right"
        else:
            return "center"
    
    def get_distance_text(self, bbox_height):
        """Estimate distance from object size"""
        if bbox_height > 300:
            return "very close"
        elif bbox_height > 150:
            return "close"
        elif bbox_height > 50:
            return "nearby"
        else:
            return "far"
    
    def run(self):
        """Main system loop"""
        if not self.camera_available:
            print("Cannot start without camera")
            return
        
        print("\nStarting detection system...")
        print(f"Detection interval: {self.config['detection_interval']} seconds")
        print("Hold objects close to camera for better detection")
        print("="*60)
        
        debug_mode = True
        last_objects_spoken = []
        
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            self.frame_count += 1
            current_time = time.time()
            display_frame = frame.copy()
            
            # Calculate FPS
            fps = self.frame_count / (current_time - self.start_time)
            
            # Run detection on interval
            if current_time - self.last_detection_time >= self.config['detection_interval']:
                print(f"\n[Detection #{int(self.frame_count/150)}]")
                
                # Run YOLO detection
                results = self.model(frame, 
                                   conf=self.config['confidence'],
                                   classes=self.config['indoor_classes'],
                                   verbose=False)[0]
                
                self.last_detection_time = current_time
                current_objects = []
                
                # Process detections
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get object name
                    if class_id in self.config['class_names']:
                        object_name = self.config['class_names'][class_id]
                    else:
                        object_name = f"object_{class_id}"
                    
                    # Calculate position and distance
                    position = self.get_position_text([x1, y1, x2, y2], frame.shape[1])
                    distance = self.get_distance_text(y2 - y1)
                    
                    # Prepare speech text
                    speech_text = f"{object_name} on your {position}, {distance}"
                    current_objects.append(speech_text)
                    
                    # Draw visualization
                    if debug_mode:
                        # Color by position
                        if position == "left":
                            color = (255, 0, 0)  # Blue
                        elif position == "right":
                            color = (0, 0, 255)  # Red
                        else:
                            color = (0, 255, 0)  # Green
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{object_name} {confidence:.2f}"
                        cv2.putText(display_frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Announce NEW objects only (not repeating same ones)
                new_objects = [obj for obj in current_objects if obj not in last_objects_spoken]
                if new_objects:
                    for obj in new_objects[:3]:  # Limit to 3 announcements at once
                        self.speech.speak(obj)
                    last_objects_spoken = current_objects.copy()
                
                print(f"   Found: {len(results.boxes)} objects")
                if len(results.boxes) > 0:
                    for i, box in enumerate(results.boxes[:5]):  # Show first 5
                        class_id = int(box.cls[0])
                        name = self.config['class_names'].get(class_id, f"obj_{class_id}")
                        conf = float(box.conf[0])
                        print(f"     {i+1}. {name} ({conf:.2f})")
            
            # Display overlay info
            time_to_next = self.config['detection_interval'] - (current_time - self.last_detection_time)
            
            # Top-left info panel
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Conf: {self.config['confidence']:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Next scan: {time_to_next:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Position guides
            cv2.line(display_frame, (int(frame.shape[1]*0.4), 0),
                    (int(frame.shape[1]*0.4), frame.shape[0]), (255, 255, 255), 1)
            cv2.line(display_frame, (int(frame.shape[1]*0.6), 0),
                    (int(frame.shape[1]*0.6), frame.shape[0]), (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Smart Glasses - Indoor Object Detection", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                self.running = False
            elif key == ord('s'):
                self.speech.speak("Speech test successful")
            elif key == ord('+'):
                self.config['confidence'] = min(0.9, self.config['confidence'] + 0.05)
                print(f"Confidence threshold: {self.config['confidence']:.2f}")
            elif key == ord('-'):
                self.config['confidence'] = max(0.1, self.config['confidence'] - 0.05)
                print(f"Confidence threshold: {self.config['confidence']:.2f}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug display: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('i'):
                self.print_indoor_classes()
    
    def print_indoor_classes(self):
        """Print all indoor classes being detected"""
        print("\n" + "="*60)
        print("INDOOR OBJECTS BEING DETECTED (39 classes):")
        print("="*60)
        
        categories = {
            "People & Pets": [0, 14, 15, 16],
            "Furniture": [56, 57, 58, 59, 60, 61],
            "Electronics": [62, 63, 64, 65, 66, 67],
            "Kitchen Appliances": [68, 69, 70, 71, 72],
            "Food & Drink": [39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            "Utensils": [42, 43, 44],
            "Stationery & Personal": [73, 74, 75, 76, 77, 78, 79],
            "Miscellaneous": [1, 13, 17, 18, 19, 20, 21, 22, 23]
        }
        
        for category, class_ids in categories.items():
            print(f"\n{category}:")
            for class_id in class_ids:
                if class_id in self.config['class_names']:
                    print(f"  {class_id:3d} - {self.config['class_names'][class_id]}")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        fps = self.frame_count / (time.time() - self.start_time)
        print(f"Total frames: {self.frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print("="*60)
        print("System shutdown complete")

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    try:
        # Create and run system
        system = SmartGlassesSystem(CONFIG)
        
        if system.camera_available:
            # Print class list at start
            system.print_indoor_classes()
            time.sleep(2)
            
            # Run main loop
            system.run()
        
        system.cleanup()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()