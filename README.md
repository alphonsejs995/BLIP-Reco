# 👓 BLIP-Powered Smart Glasses: Visual Intelligence System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time visual intelligence system for smart glasses using **BLIP (Bootstrapping Language-Image Pre-training)** for scene understanding, object description, and visual question answering.

## 🚀 Features

- **Real-time Image Captioning**: Generate natural language descriptions of the environment
- **Visual Question Answering**: Ask questions about the scene ("What color is the car?")
- **Object Contextualization**: Beyond detection - understand relationships and attributes
- **Lightweight Integration**: Optimized for edge devices (glasses/AR wearables)
- **Modular Architecture**: Easy to extend with custom vision/language models

## 📁 Project Structure
BLIP-Reco/
├── pyt_blip.py # Main BLIP implementation
├── requirements.txt # Python dependencies
├── .gitignore # Git exclusion rules
├── README.md # This file
└── examples/ # (Optional) Example usage scripts
├── test_caption.py
└── test_vqa.py


## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/alphonsejs995/BLIP-Reco.git
cd BLIP-Reco
```
### 2. Create Virtual Environment
```bash
python -m venv glasses_env
```
#### Windows:
```
glasses_env\Scripts\activate
```
#### Mac/Linux:
```
source glasses_env/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 🎯 Quick Start
## Basic Image Captioning
```python
from pyt_blip import BLIPModel

# Initialize model
blip = BLIPModel(model_type="captioning")

# Generate caption for an image
caption = blip.generate_caption("scene.jpg")
print(f"Scene: {caption}")
# Output: "a person wearing smart glasses looking at a computer screen"
```
## Visual Question Answering
```python
from pyt_blip import BLIPModel

# Initialize VQA model
blip_vqa = BLIPModel(model_type="vqa")

# Ask about an image
answer = blip_vqa.ask_question(
    image_path="street.jpg",
    question="Is there a pedestrian crossing?"
)
print(f"Answer: {answer}")
# Output: "yes, there is a zebra crossing"
```
## 🔧 API Reference

### BLIPModel Class
```python
class BLIPModel:
    def __init__(self, model_type="captioning", device="auto"):
        """
        Initialize BLIP model.
        
        Args:
            model_type: "captioning" or "vqa"
            device: "cuda", "cpu", or "auto"
        """
    
    def generate_caption(self, image_path, prompt=None):
        """
        Generate caption for an image.
        
        Args:
            image_path: Path to image file
            prompt: Optional context prompt
        Returns:
            str: Generated caption
        """
    
    def ask_question(self, image_path, question):
        """
        Answer a question about an image.
        
        Args:
            image_path: Path to image file
            question: Question string
        Returns:
            str: Answer
        """
```
### 🤖 Integration with Smart Glasses
#### Example: Real-time Scene Analysis
```python
import cv2
from pyt_blip import BLIPModel

class SmartGlassesSystem:
    def __init__(self):
        self.blip = BLIPModel()
        self.camera = cv2.VideoCapture(0)  # Glasses camera
    
    def analyze_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Save temporary frame
            cv2.imwrite("temp_frame.jpg", frame)
            # Get scene description
            description = self.blip.generate_caption("temp_frame.jpg")
            return description
        return None
    
    def run(self):
        while True:
            desc = self.analyze_frame()
            if desc:
                print(f"Current scene: {desc}")
                # Send to glasses display/speaker
                self.display_output(desc)
```

## 🌟 Use Cases
Assistive Technology: Help visually impaired users understand their environment
AR Navigation: Provide contextual information during navigation
Educational Tool: Real-time object identification and explanation
Security: Scene monitoring with natural language alerts
Research: Multimodal AI experimentation platform

## 🛠️ Customization
Fine-tuning on Custom Data
```python
# Example: Fine-tune for specific objects
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Add your training loop here
# (See Hugging Face documentation for full fine-tuning guide)
```
## 📈 Future Enhancements
BLIP-2 integration for improved performance
Whisper integration for voice commands
Edge optimization with ONNX/TensorRT
Multi-language support
Real-time video streaming analysis

## 🤝 Contributing
Contributions welcome! Please:
Fork the repository
Create a feature branch
Submit a Pull Request

## 📄 License
MIT License - see LICENSE file for details.

## 🙏 Acknowledgments
Salesforce Research for BLIP
Hugging Face for Transformers library
PyTorch team for the deep learning framework

## 📧 Contact
Alphonse J S

GitHub: @alphonsejs995

Project Link: https://github.com/alphonsejs995/BLIP-Reco
