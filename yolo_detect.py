from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
# model = YOLO('yolov8n.pt')
# model=YOLO('runs/detect/new_train/train/weights/best.pt')
model=YOLO('runs/detect/new_train/train/weights/best.pt')

# Run detection on an image
results=model('datasets/Cans detection/test/images/youtube-1_jpg.rf.f2ded39edf3082ce5b487a24bb6dabbc.jpg') # testing on test data
results1=model('pepsi.jpg') # testing on unseen data


def count_classes(results):
    for r in results:
        r.show()
        class_indices=r.boxes.cls.int().tolist()
        class_names=r.names
        print(class_names[2])
        counts={name:0 for name in class_names.values()}
        print(counts)
        for idx in class_indices:
            counts[class_names[idx]]+=1

        for name, count in counts.items():
            print(f"Number of {name} detected: {count}")

count_classes(results)            

# model.predict('test/images/youtube-110_jpg.rf.eab8fb76d0d8ad0fce3fe657716c7a45.jpg', save=True)
# model.predict('OIP (1).jpg', save=True)


