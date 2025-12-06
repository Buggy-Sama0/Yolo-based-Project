# YOLO-based Object Detection Project

## Project Structure


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

## Customization

## Evaluation



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

