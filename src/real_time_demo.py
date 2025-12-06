import cv2
from ultralytics import YOLO

class RealTimeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow('YOLO Real-time Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeDetector('models/runs/detect/fine-tuned-yolo/train/weights/best.pt')
    detector.run()
