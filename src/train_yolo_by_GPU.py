'''
Fine-tuning the Pretrained YOLO Model using the GPU
This section demonstrates how to fine-tune a pretrained YOLO model 
using your custom dataset.
'''
import multiprocessing
import torch
from ultralytics import YOLO


def main():
	# use GPU if available, otherwise CPU
	device = 0 if torch.cuda.is_available() else 'cpu'

	# Load a pre-trained model
	model = YOLO('yolov8n.pt')

	# Fine-tune the model
	model.train(
		data='data.yaml',
		# Training schedule
		epochs=50,           
		imgsz=640,
		batch=8,
		device=device,
		workers=4,
		half=True,
		augment=True,
		patience=10,
		lr0=0.001,
		project='runs/detect/fine-tuned-yolo',
	)


if __name__ == '__main__':
	multiprocessing.freeze_support()
	main()




