'''
Fine-tuning the Pretrained YOLO Model using the GPU
This section demonstrates how to fine-tune a pretrained YOLO model 
using your custom dataset.
'''
import multiprocessing
import torch
from ultralytics import YOLO


def main():
	# Auto-select device: use GPU if available, otherwise CPU
	device = 0 if torch.cuda.is_available() else 'cpu'

	# Load a pre-trained model
	model = YOLO('yolov8n.pt')

	# Fine-tune the model
	model.train(
		data='./config/data.yaml',
		# Training schedule
		epochs=50,           
		# Input size and batch tuned for higher accuracy on GPU
		imgsz=640,
		batch=8,
		# Device (0 for first GPU or 'cpu')
		device=device,
		# DataLoader workers and mixed precision for faster training on GPU
		workers=4,
		half=True,
		# Enable basic augmentation and early stopping patience
		augment=True,
		patience=10,
		# Learning rate (start conservative)
		lr0=0.001,
		project='runs/detect/fine-tuned-yolo',
	)


if __name__ == '__main__':
	# On Windows, multiprocessing with spawn requires freeze_support()
	multiprocessing.freeze_support()
	main()




