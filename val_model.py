from ultralytics import YOLO

model=YOLO('runs/detect/train1/weights/best.pt')
metrics=model.val()
print(metrics.box.map)