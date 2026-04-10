# EcoVision: Technical Specification & Innovation Profile

## 1. Project Background
Effective waste management starts with precise identification. While traditional systems can detect broad categories like "Bottle," they often fail to distinguish between different materials (Glass vs. Plastic) or shapes (Disposable vs. Handle Cup). This distinction is vital for high-accuracy sorting in recycling facilities. **EcoVision** utilizes Deep Learning to provide this granular level of categorization.

## 2. Theoretical Framework
The system implements a **YOLOv8n (Nano)** model. This single-stage detector performs bounding box regression and class prediction simultaneously, making it suitable for high-speed edge computing.

### Key Performance Metrics (Targeted):
- **mAP@0.5:** > 0.85
- **Inference Speed:** < 20ms per frame (on NVIDIA T4 or better)
- **Dataset Size:** ~1,200 annotated images across 9 classes

## 3. System Architecture
### Hardware Layer
- Compatible with Standard Webcams, Smartphone Cameras, or Industrial Industrial Cameras.
- Optimization for CUDA/cuDNN on NVIDIA GPUs and MPS for Apple Silicon.

### AI Engine Layer
- **Input Preprocessing:** Auto-orientation, resizing to 640px, and normalization.
- **Inference Engine:** Ultralytics YOLO inference pipeline.
- **Post-processing:** Non-Maximum Suppression (NMS) with a threshold of 0.45 and Confidence threshold of 0.25.

### Presentation Layer
- **Web Dashboard:** Built with Flask, serving as a Remote Monitoring System (RMS).
- **Socket Connectivity:** Potential for real-time WebSocket communication of detection events (Future development).

## 4. Competitive Advantages
1. **Material Sensitivity:** Capable of classifying objects based on material (e.g., distinguishing between a paper 'disposable cup' and a 'glass mug').
2. **Contextual Recognition:** Recognizes 'gym bottles' as a distinct class, which are often multi-material and difficult for standard recycling scanners.
3. **Deployment Versatility:** The small footprint of the YOLOv8n model allows for deployment on low-power devices like Raspberry Pi 5.

## 5. Potential Social Impact
EcoVision aligns with the **United Nations Sustainable Development Goal 12: Responsible Consumption and Production**. By integrating our model into Smart Bins, we can reduce waste contamination by up to 30% through immediate user feedback via visual detection.
