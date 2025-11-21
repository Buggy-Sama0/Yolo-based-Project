# YOLO-based Object Detection Project

## Project Structure

- `images/train`, `images/val`: Place your training and validation images here.
- `labels/train`, `labels/val`: Place your YOLO-format label files here.
- `data.yaml`: Dataset and augmentation configuration.
- `requirements.txt`: Python dependencies.
- `quick_test.py`: Run a quick YOLOv8 test.
- `train_yolo.py`: Script to train YOLOv8 on your dataset.
- `real_time_demo.py`: Real-time webcam detection demo.
- `custom_augmentation.py`: Custom data augmentation pipeline.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your dataset in YOLO format (see `data.yaml`).
3. Run a quick YOLO test:
   ```bash
   python quick_test.py
   ```
4. Train your model:
   ```bash
   python train_yolo.py
   ```
5. Try the real-time webcam demo:
   ```bash
   python real_time_demo.py
   ```

## Dataset Conversion
- Convert your Mask R-CNN/COCO dataset to YOLO format using Roboflow or a custom script.

## Customization
- Edit `custom_augmentation.py` to add your own augmentations.
- Adjust `data.yaml` for your classes and augmentation settings.

## Evaluation
- Compare YOLO results with your Mask R-CNN baseline using mAP, precision, recall, and FPS.



results = model.train(
    data='data.yaml',
    epochs=20,
    imgsz=320,
    batch=4,
    device='cpu',
    patience=5,
    lr0=0.01,
    optimizer='Adam',
    weight_decay=0.0005,
    cache=True,
    workers=1,
    project='runs/detect/new_train',
    save=True
)

