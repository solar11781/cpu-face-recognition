# Face Recognition Attendance System with Anti-Spoofing

## Overview
This project implements a real-time Face Recognition Attendance System designed to enhance security and efficiency for tasks such as attendance tracking and access control. The system integrates face detection, recognition, and anti-spoofing capabilities into a lightweight, CPU-friendly solution suitable for real-time applications.

Key features include:
- Multi-angle face registration.
- Real-time face detection using lightweight models.
- Face recognition with robust feature extraction.
- Anti-spoofing to detect and block fake faces from photos or videos.
- Experiments with GAN-based image enhancement to improve recognition under challenging conditions.

---

## Features
### Face Registration
- Supports capturing images from multiple face orientations (front, up, down, left, right).
- Images are processed and stored for training and recognition.

### Face Detection
- Utilizes the SSD ResNet-10 model for efficient and accurate face detection.
- Optimized for CPU-only environments.

### Face Recognition
- Employs the VGG-Face model for feature extraction and recognition.
- Uses MTCNN for facial landmark detection and bounding box generation.
- Capable of recognizing faces even with accessories like masks or hats.

### Anti-Spoofing
- Integrates the Silent Face Anti-Spoofing framework to block spoofed inputs (e.g., photos or videos).
- Analyzes texture, depth, and motion inconsistencies to identify fake faces.

### GAN-Based Image Enhancement
- Experiments with Real-ESRGAN and GFPGAN to enhance image quality in challenging scenarios (e.g., low light or motion blur).

---

## Technologies Used
- **Programming Languages:** Python
- **Libraries and Frameworks:**
  - TensorFlow, PyTorch, OpenCV
  - DeepFace for face recognition
  - Silent Face Anti-Spoofing framework
- **GAN Models:**
  - Real-ESRGAN and GFPGAN for image enhancement

---

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nguyengiabinh/CPU-only-Face-recognition.git

   pip install -r requirements.txt

   python app.py



