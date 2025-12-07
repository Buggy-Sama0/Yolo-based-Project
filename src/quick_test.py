from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.predict('', save=True)
print("✅ YOLO is working!")

