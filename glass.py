"""
Smart AI Glasses - Complete System (FIXED VERSION)
Run: python smart_ai_glasses.py
Press 'q' to quit, 't' to test speech, 'o' to test OCR
"""

import cv2
import time
import numpy as np
import threading
import queue
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import json

print("=== Smart AI Glasses System ===")
print("Initializing...")

# =================== CONFIGURATION ===================
@dataclass
class Config:
    """All configuration in one place"""
    # Detection
    DETECTION_MODEL: str = "yolov8n.pt"
    DETECTION_CONFIDENCE: float = 0.3  # Lowered for better detection
    DETECTION_CLASSES: List[int] = None
    
    # Camera
    CAMERA_ID: int = 0  # 0 = default webcam
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 10
    
    # Speech
    SPEECH_RATE: int = 150
    SPEECH_VOLUME: float = 1.0
    
    # OCR
    OCR_LANGUAGE: str = "en"
    OCR_CONFIDENCE: float = 0.4
    
    # Timing
    OBJECT_INTERVAL: float = 1.0  # Detect objects every 1 sec (faster)
    TEXT_INTERVAL: float = 3.0    # Detect text every 3 sec
    MIN_REPEAT_TIME: float = 10.0 # Don't repeat same object within 10 sec
    
    # Display
    SHOW_FPS: bool = True
    SHOW_BBOX: bool = True
    DEBUG_MODE: bool = True  # Added for debug prints
    
    def __post_init__(self):
        if self.DETECTION_CLASSES is None:
            self.DETECTION_CLASSES = [
                0,  # person
                1,  # bicycle
                2,  # car
                3,  # motorcycle
                13, # bench
                14, # bird
                15, # cat
                16, # dog
                39, # bottle
                56, # chair
                57, # couch
                60, # dining table
                62, # tv
                63, # laptop
                67, # cell phone
                73, # book
            ]
        
        # Object sizes for distance estimation (in meters)
        self.OBJECT_SIZES = {
            'person': 1.7, 'chair': 0.8, 'laptop': 0.3,
            'bottle': 0.25, 'book': 0.25, 'cell phone': 0.15,
            'car': 1.5, 'cup': 0.1, 'keyboard': 0.05,
            'dog': 0.5, 'cat': 0.3, 'tv': 1.0
        }

# Initialize config
config = Config()

# =================== SPEECH MANAGER ===================
class SpeechManager:
    """Handles text-to-speech with queuing"""
    
    def __init__(self, config):
        self.config = config
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_history = {}  # For deduplication: {object_id: last_spoken_time}
        
        # Try to initialize pyttsx3
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', config.SPEECH_RATE)
            self.engine.setProperty('volume', config.SPEECH_VOLUME)
            self.tts_available = True
            print("✓ Speech engine initialized (pyttsx3)")
        except Exception as e:
            print(f"✗ Speech engine failed: {e}")
            print("Will display text instead of speaking")
            self.tts_available = False
        
        # Start speech thread
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()
    
    def speak(self, text, object_id=None):
        """Add speech to queue with deduplication"""
        current_time = time.time()
        
        # Deduplication logic
        if object_id:
            if object_id in self.speech_history:
                time_since_last = current_time - self.speech_history[object_id]
                if time_since_last < config.MIN_REPEAT_TIME:
                    if config.DEBUG_MODE:
                        print(f"  Skipping duplicate: {object_id}")
                    return
            
            self.speech_history[object_id] = current_time
        
        # Add to queue
        self.speech_queue.put(text)
        print(f"📢 Queued: {text}")
    
    def _speech_worker(self):
        """Background thread that processes speech queue"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                self.is_speaking = True
                
                if self.tts_available:
                    if config.DEBUG_MODE:
                        print(f"  Speaking: {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print(f"[SPEECH]: {text}")
                
                self.is_speaking = False
                self.speech_queue.task_done()
                
                # Small pause between speeches
                time.sleep(0.3)
                
            except queue.Empty:
                continue
    
    def test_speech(self):
        """Test if speech is working"""
        test_text = "Smart glasses system is working"
        self.speak(test_text, "test")
        return True

# =================== OBJECT DETECTOR ===================
class ObjectDetector:
    """Handles object detection with YOLOv8"""
    
    def __init__(self, config):
        self.config = config
        
        print("Loading YOLOv8 model...")
        try:
            from ultralytics import YOLO
            # This will download model if not present
            self.model = YOLO(config.DETECTION_MODEL)
            self.model_loaded = True
            print(f"✓ YOLOv8 model loaded: {config.DETECTION_MODEL}")
            
            # Warm-up inference
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False)
            
        except ImportError:
            print("✗ ultralytics not installed. Installing...")
            os.system(f"{sys.executable} -m pip install ultralytics")
            from ultralytics import YOLO
            self.model = YOLO(config.DETECTION_MODEL)
            self.model_loaded = True
            print("✓ YOLOv8 installed and loaded")
        except Exception as e:
            print(f"✗ Failed to load YOLOv8: {e}")
            self.model_loaded = False
    
    def detect(self, frame):
        """Detect objects in frame"""
        if not self.model_loaded:
            return [], frame
        
        try:
            # Run inference
            results = self.model(frame, 
                                conf=self.config.DETECTION_CONFIDENCE,
                                classes=self.config.DETECTION_CLASSES,
                                verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                detections.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id
                })
            
            # Debug print
            if config.DEBUG_MODE and detections:
                print(f"  YOLO found {len(detections)} objects")
                for det in detections[:3]:  # Show first 3
                    print(f"    - {det['name']} ({det['confidence']:.2f})")
            
            return detections, frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], frame

# =================== OCR READER ===================
class OCRReader:
    """Handles text detection and recognition"""
    
    def __init__(self, config):
        self.config = config
        self.reader = None
        
        print("Loading OCR model...")
        try:
            import easyocr
            # Use GPU if available (on laptop), CPU on Pi
            try:
                self.reader = easyocr.Reader([config.OCR_LANGUAGE], gpu=True)
                print("✓ OCR loaded with GPU support")
            except:
                # Fallback to CPU
                self.reader = easyocr.Reader([config.OCR_LANGUAGE], gpu=False)
                print("✓ OCR loaded with CPU (GPU not available)")
            
            self.ocr_loaded = True
            
        except ImportError:
            print("✗ easyocr not installed. Will skip OCR functionality.")
            self.ocr_loaded = False
        except Exception as e:
            print(f"✗ OCR loading error: {e}")
            self.ocr_loaded = False
    
    def read_text(self, frame):
        """Extract text from frame"""
        if not self.ocr_loaded or self.reader is None:
            return [], frame
        
        try:
            # Convert to RGB for EasyOCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            results = self.reader.readtext(rgb_frame)
            
            texts = []
            for (bbox, text, confidence) in results:
                if confidence >= self.config.OCR_CONFIDENCE:
                    texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            # Debug print
            if config.DEBUG_MODE and texts:
                print(f"  OCR found {len(texts)} text regions")
                for txt in texts[:2]:
                    print(f"    - '{txt['text'][:30]}' ({txt['confidence']:.2f})")
            
            return texts, frame
            
        except Exception as e:
            print(f"OCR error: {e}")
            return [], frame

# =================== VISUALIZER ===================
class Visualizer:
    """Handles display and visualization"""
    
    @staticmethod
    def draw_object(frame, detection, spatial_info):
        """Draw object bounding box and info"""
        x1, y1, x2, y2 = map(int, detection['bbox'])
        
        # Choose color based on position
        if spatial_info['position'] == 'left':
            color = (255, 0, 0)  # Blue
        elif spatial_info['position'] == 'right':
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"{detection['name']} {spatial_info['position']}"
        if spatial_info['distance']:
            label += f" {spatial_info['distance']}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1-text_size[1]-5), 
                     (x1+text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def draw_text_region(frame, text_detection):
        """Draw text bounding box"""
        bbox = text_detection['bbox']
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw polygon for text region
        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)  # Cyan
        
        # Draw text label
        text_preview = text_detection['text'][:30] + ("..." if len(text_detection['text']) > 30 else "")
        label = f"Text: {text_preview}"
        
        # Find top-left point for label
        min_x = min([p[0] for p in bbox])
        min_y = min([p[1] for p in bbox])
        
        cv2.putText(frame, label, (min_x, min_y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    @staticmethod
    def draw_debug_info(frame, fps, speech_status, obj_count, text_count):
        """Draw debug information on frame"""
        # Draw FPS
        if config.SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw speech status
        status_color = (0, 0, 255) if speech_status else (0, 255, 0)
        status_text = "SPEAKING" if speech_status else "READY"
        cv2.putText(frame, f"Status: {status_text}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw counters
        cv2.putText(frame, f"Objects: {obj_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Texts: {text_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw position zones
        height, width = frame.shape[:2]
        cv2.line(frame, (int(width*0.4), 0), (int(width*0.4), height), 
                (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (int(width*0.6), 0), (int(width*0.6), height), 
                (255, 255, 255), 1, cv2.LINE_AA)
        
        # Zone labels
        cv2.putText(frame, "LEFT", (int(width*0.1), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "CENTER", (int(width*0.45), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "RIGHT", (int(width*0.75), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

# =================== SPATIAL ANALYZER ===================
class SpatialAnalyzer:
    """Calculates position and distance of objects"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze(self, detection, frame_width):
        """Calculate position and distance for an object"""
        x1, y1, x2, y2 = detection['bbox']
        
        # Calculate center
        center_x = (x1 + x2) / 2
        
        # Determine position
        if center_x < frame_width * 0.4:
            position = "left"
        elif center_x > frame_width * 0.6:
            position = "right"
        else:
            position = "center"
        
        # Estimate distance
        distance = None
        object_name = detection['name']
        
        if object_name in self.config.OBJECT_SIZES:
            real_height = self.config.OBJECT_SIZES[object_name]
            bbox_height = y2 - y1
            
            if bbox_height > 0:
                # Simplified distance estimation
                # Assuming camera FOV and object size relationship
                if bbox_height > 300:
                    distance = "very close"
                elif bbox_height > 150:
                    distance = "close"
                elif bbox_height > 50:
                    distance = "moderate"
                else:
                    distance = "far"
        
        return {
            'position': position,
            'distance': distance,
            'center_x': center_x
        }

# =================== MAIN SYSTEM ===================
class SmartGlassesSystem:
    """Main system that integrates all components"""
    
    def __init__(self, config):
        self.config = config
        
        print("\n" + "="*50)
        print("INITIALIZING SMART GLASSES SYSTEM")
        print("="*50)
        
        # Initialize components
        self.speech = SpeechManager(config)
        self.detector = ObjectDetector(config)
        self.ocr = OCRReader(config)
        self.visualizer = Visualizer()
        self.spatial = SpatialAnalyzer(config)
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        if not self.cap.isOpened():
            print(f"✗ Cannot open camera {config.CAMERA_ID}")
            print("Trying camera 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("✗ No camera found!")
                self.camera_available = False
                return
        
        self.camera_available = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        print(f"✓ Camera initialized: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        
        # State tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_object_time = 0
        self.last_text_time = 0
        self.running = True
        
        print("\n" + "="*50)
        print("SYSTEM READY")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  't' - Test speech")
        print("  'o' - Test OCR on current frame")
        print("  's' - Save screenshot")
        print("  'c' - Clear speech queue")
        print("  'd' - Toggle debug mode")
        print("="*50 + "\n")
    
    def generate_speech_text(self, detection, spatial_info):
        """Generate natural language description"""
        object_name = detection['name']
        position = spatial_info['position']
        distance = spatial_info['distance']
        
        # Position mapping to natural language
        position_map = {
            'left': 'on your left',
            'right': 'on your right',
            'center': 'in front of you'
        }
        
        position_text = position_map.get(position, position)
        
        if distance:
            return f"{object_name} {position_text}, {distance}"
        else:
            return f"{object_name} {position_text}"
    
    def run(self):
        """Main system loop - FIXED VERSION"""
        if not self.camera_available:
            print("Cannot run without camera. Exiting.")
            return
        
        print("Starting main loop...")
        
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Create a fresh display frame for visualization
            display_frame = frame.copy()
            
            self.frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Initialize counters
            object_count = 0
            text_count = 0
            
            # === OBJECT DETECTION ===
            if current_time - self.last_object_time >= self.config.OBJECT_INTERVAL:
                detections, _ = self.detector.detect(frame)
                self.last_object_time = current_time
                
                # Process each detection
                for detection in detections:
                    if detection['confidence'] > 0.5:  # Slightly lowered threshold
                        # Calculate spatial information
                        spatial_info = self.spatial.analyze(detection, display_frame.shape[1])
                        
                        # Generate speech
                        speech_text = self.generate_speech_text(detection, spatial_info)
                        object_id = f"{detection['name']}_{spatial_info['position']}"
                        
                        # Speak
                        self.speech.speak(speech_text, object_id)
                        
                        # Visualize on DISPLAY frame (FIXED)
                        if self.config.SHOW_BBOX:
                            display_frame = self.visualizer.draw_object(display_frame, detection, spatial_info)
                        
                        object_count += 1
            
            # === TEXT DETECTION ===
            if current_time - self.last_text_time >= self.config.TEXT_INTERVAL:
                texts, _ = self.ocr.read_text(frame)
                self.last_text_time = current_time
                
                # Process detected text
                if texts:
                    # Find the most confident text
                    best_text = max(texts, key=lambda x: x['confidence'])
                    
                    if best_text['confidence'] > 0.5:  # Lowered threshold
                        # Generate speech
                        text_preview = best_text['text'][:50]
                        speech_text = f"Text detected: {text_preview}"
                        text_id = f"text_{hash(text_preview) % 1000}"  # Simple ID
                        
                        # Speak
                        self.speech.speak(speech_text, text_id)
                        
                        # Visualize
                        if self.config.SHOW_BBOX:
                            display_frame = self.visualizer.draw_text_region(display_frame, best_text)
                        
                        text_count = len(texts)
            
            # === DISPLAY ===
            # Add debug information to DISPLAY frame (FIXED)
            display_frame = self.visualizer.draw_debug_info(
                display_frame, fps, 
                self.speech.is_speaking,
                object_count, text_count
            )
            
            # Show the DISPLAY frame (with all drawings) - FIXED
            cv2.imshow("Smart AI Glasses", display_frame)
            
            # === KEYBOARD CONTROLS ===
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                self.running = False
                break
                
            elif key == ord('t'):
                print("Testing speech...")
                self.speech.test_speech()
                
            elif key == ord('o'):
                print("Testing OCR on current frame...")
                texts, _ = self.ocr.read_text(frame)
                if texts:
                    print(f"Found {len(texts)} text regions:")
                    for i, text in enumerate(texts):
                        print(f"  {i+1}. '{text['text']}' (conf: {text['confidence']:.2f})")
                else:
                    print("No text found in current frame")
                    
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)  # Save display frame
                print(f"Screenshot saved: {filename}")
                
            elif key == ord('c'):
                # Clear speech queue
                while not self.speech.speech_queue.empty():
                    try:
                        self.speech.speech_queue.get_nowait()
                        self.speech.speech_queue.task_done()
                    except queue.Empty:
                        break
                print("Speech queue cleared")
                
            elif key == ord('d'):
                # Toggle debug mode
                config.DEBUG_MODE = not config.DEBUG_MODE
                print(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")
    
    def test_all_components(self):
        """Test all system components"""
        print("\n" + "="*50)
        print("COMPONENT TESTING")
        print("="*50)
        
        # Test 1: Camera
        print("1. Testing camera...")
        ret, frame = self.cap.read()
        if ret:
            print(f"   ✓ Camera working ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print("   ✗ Camera failed")
            return False
        
        # Test 2: Object Detection
        print("2. Testing object detection...")
        if self.detector.model_loaded:
            detections, _ = self.detector.detect(frame)
            print(f"   ✓ Detection working. Found {len(detections)} objects")
            if detections:
                for det in detections[:5]:  # Show first 5
                    print(f"     - {det['name']} ({det['confidence']:.2f})")
            else:
                print("   ⚠ No objects detected in test frame")
        else:
            print("   ✗ Detection failed")
        
        # Test 3: Speech
        print("3. Testing speech...")
        self.speech.test_speech()
        print("   ✓ Speech test initiated")
        
        # Test 4: OCR
        print("4. Testing OCR...")
        if self.ocr.ocr_loaded:
            texts, _ = self.ocr.read_text(frame)
            print(f"   ✓ OCR working. Found {len(texts)} text regions")
            if texts:
                for text in texts[:3]:  # Show first 3
                    print(f"     - '{text['text']}' ({text['confidence']:.2f})")
            else:
                print("   ⚠ No text found in test frame")
        else:
            print("   ⚠ OCR not loaded (optional)")
        
        print("="*50)
        print("Testing complete. Starting main system...")
        print("="*50 + "\n")
        
        return True

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    # Install missing packages automatically
    def install_package(package):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Check and install required packages
    required_packages = ['opencv-python', 'numpy']
    
    print("Checking dependencies...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            install_package(package)
    
    # Create system
    system = SmartGlassesSystem(config)
    
    if system.camera_available:
        # Run component tests
        system.test_all_components()
        
        # Wait a moment for speech test to complete
        time.sleep(2)
        
        # Start main loop
        try:
            system.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("System cannot start without camera.")
    
    print("\n" + "="*50)
    print("SYSTEM SHUTDOWN")
    print("="*50)