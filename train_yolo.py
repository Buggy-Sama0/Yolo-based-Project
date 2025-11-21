'''
Fine-tuning the Pretrained YOLO Model
This section demonstrates how to fine-tune a pretrained YOLO model 
using your custom dataset.
'''
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Fine-tune the model
model.train(
    data='data.yaml',
    epochs=50,           # Set the number of epochs as needed
    imgsz=320,           # Image size
    batch=4,            # Batch size
    lr0=0.001,           # Initial learning rate
    device='cpu',             # Use GPU if available
    project='runs/detect/trained_model',
)

# Validate on training data
model.val()




