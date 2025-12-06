from ultralytics import YOLO

model=YOLO('models/runs/detect/fine-tuned-yolo/train/weights/best.pt')
metrics=model.val()
print(metrics.box.map)