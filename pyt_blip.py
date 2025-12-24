"""
SMART GLASSES - BLIP DESCRIPTOR
Describes ANYTHING in the scene naturally
"""

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

print("="*60)
print("SMART GLASSES - NATURAL SCENE DESCRIPTOR")
print("Uses BLIP to describe ANY object naturally")
print("="*60)

class SceneDescriber:
    def __init__(self):
        print("Loading BLIP model (first time: ~1.5GB download)...")
        
        # Load BLIP model - describes images with natural language
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        
        print("✓ BLIP model loaded")
        print("It will describe anything it sees naturally")
    
    def describe_scene(self, frame):
        """Generate natural description of what's in the frame"""
        # Convert OpenCV frame to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        
        # Generate description
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=50)
        
        # Decode to text
        description = self.processor.decode(output[0], skip_special_tokens=True)
        return description

# =================== MAIN SYSTEM ===================
def main():
    # Initialize
    describer = SceneDescriber()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Camera failed")
        return
    
    print("\n" + "="*60)
    print("Press 'd' to describe scene")
    print("Press 'q' to quit")
    print("Show your fan, treadmill, chair - then press 'd'")
    print("="*60)
    
    last_description_time = 0
    description_interval = 5.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'd' to describe scene", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Smart Glasses - BLIP Descriptor", display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            current_time = time.time()
            if current_time - last_description_time >= description_interval:
                print("\n" + "-"*40)
                print("Analyzing scene...")
                
                description = describer.describe_scene(frame)
                print(f"DESCRIPTION: {description}")
                
                last_description_time = current_time
            else:
                print(f"Wait {description_interval - (current_time - last_description_time):.1f}s")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Install first: pip install transformers torch torchvision
    main()