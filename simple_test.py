# save as test_install.py
import cv2
import numpy as np
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

try:
    from ultralytics import YOLO
    print("✓ YOLO installed")
except ImportError as e:
    print(f"✗ YOLO error: {e}")

try:
    import easyocr
    print("✓ EasyOCR installed")
except ImportError as e:
    print(f"✗ EasyOCR error: {e}")

try:
    import pyttsx3
    print("✓ pyttsx3 installed")
except ImportError as e:
    print(f"✗ pyttsx3 error: {e}")