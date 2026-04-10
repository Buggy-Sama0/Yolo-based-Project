# EcoVision: A Real-time Deep Learning Framework for Automated Beverage Container Classification

## 🌟 Overview
**EcoVision** is an innovative computer vision solution designed to automate the classification of beverage containers. Leveraging the state-of-the-art **YOLOv8** (You Only Look Once) architecture, this project addresses the critical need for efficient waste sorting and inventory management in hospitality, retail, and recycling industries.

By distinguishing between nine distinct categories of containers (glass, plastic, metal, and ceramic), EcoVision provides a high-speed, high-accuracy alternative to manual identification.

---

## 🚀 Key Features
- **Real-time Detection:** Process video streams at 30+ FPS for instantaneous feedback.
- **Granular Classification:** Specifically fine-tuned to differentiate between similar objects like glass mugs vs. normal glasses.
- **Interactive Dashboard:** A Flask-based web interface for easy deployment and testing.
- **GPU Optimized:** Full support for CUDA-enabled training and inference for maximum performance.
- **Industry Ready:** Scalable architecture suitable for integration into smart bins or automated checkout systems.

---

## 🛠️ Technical Specifications
### 1. Model Architecture
- **Base Model:** Ultralytics YOLOv8n (Nano) - chosen for its optimal balance between speed and accuracy.
- **Customization:** Fine-tuned on a specialized dataset of beverage containers.
- **Input Resolution:** 640x640 pixels.

### 2. Software Stack
- **Language:** Python 3.12+
- **Deep Learning:** PyTorch & Ultralytics
- **Web Interface:** Flask (Backend) + HTML5/CSS3 (Frontend)
- **Computer Vision:** OpenCV (Open Source Computer Vision Library)

### 3. Classification Classes
The model is trained to recognize 9 specific object types:
1. `bottle-glass`
2. `bottle-plastic`
3. `cup-disposable`
4. `cup-handle`
5. `glass-mug`
6. `glass-normal`
7. `glass-wine`
8. `gym bottle`
9. `tin can`

---

## 📦 Project Structure
- `src/dashboard/`: Contains the Flask web application.
- `src/real_time_demo.py`: Script for webcam-based live inference.
- `models/`: Stores the fine-tuned weights (`best.pt`).
- `datasets/`: Configuration and data mapping files.
- `notebooks/`: Experimental results and data exploration.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.12 installed.
- (Optional) NVIDIA GPU with CUDA for faster performance.

### Step 1: Clone and Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd eco-vision

# Create a virtual environment
python -m venv yolo-env
source yolo-env/bin/activate  # On Windows: yolo-env\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage Guide

### Option A: Web-Based Dashboard
Launch the interactive interface to upload and analyze images or view live streams.
```bash
python src/dashboard/app.py
```
*Access via: http://127.0.0.1:5000*

### Option B: Real-time Demo (Webcam)
Run the script for immediate local camera inference.
```bash
python src/real_time_demo.py
```

---

## 📊 Innovation Highlights
Unlike standard object detectors, **EcoVision** specializes in "Sub-Class Distinction." Most models treat all containers as "bottles." EcoVision successfully differentiates between **Materials** (Glass vs. Plastic) and **Usage Contexts** (Disposable vs. Handle Cups), which is vital for high-quality recycling protocols.

---

## 🎓 Competition Context
**Project Name:** EcoVision  
**Project Category:** Information Technology / Innovative Inventions  
**Target:** Smart Waste Management & Automated Retail  
**Institution:** [Your University Name]



