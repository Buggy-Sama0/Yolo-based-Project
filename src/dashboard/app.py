
from flask import Response
from flask import Flask, render_template, request, url_for, send_from_directory
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__, static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../media')))

# Load YOLO model once
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/runs/detect/fine-tuned-yolo/train/weights/best.pt'))
if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)
else:
    yolo_model = None
    print(f'No such file or directory {MODEL_PATH}')

@app.route('/')
def index():
    image_folder = app.static_folder
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', images=images)


# Existing image detection route
@app.route('/detect', methods=['POST'])
def detect():
    image_name = request.form.get('image')
    image_path = os.path.join(app.static_folder, image_name)
    import io, base64
    if yolo_model and os.path.exists(image_path):
        results = yolo_model(image_path)
        product_counts = {}
        img_b64 = None
        for r in results:
            image = cv2.imread(image_path)
            class_indices = r.boxes.cls.int().tolist()
            class_names = r.names
            product_counts = {name: 0 for name in class_names.values()}
            for idx in class_indices:
                product_counts[class_names[idx]] += 1
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    class_id = box.cls[0].int().item()
                    class_name = r.names[class_id]
                    confidence = box.conf[0].item()
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset = 30
            for name, count in product_counts.items():
                cv2.putText(image, f"{name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 30
            # Encode image as base64 for HTML display
            _, buffer = cv2.imencode('.png', image)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
        result_msg = f"Detection completed for {image_name}."
    else:
        result_msg = "Model or image not found."
        img_b64 = None
        product_counts = {}
    return render_template('result.html', result=result_msg, image_b64=img_b64, product_counts=product_counts)

# New upload route for images/videos
@app.route('/upload', methods=['POST'])
def upload():
    import io, base64
    from werkzeug.utils import secure_filename
    file = request.files.get('file')
    if not file:
        return render_template('result.html', result='No file uploaded.', image_b64=None, product_counts={})
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    temp_path = os.path.join('temp_uploads', filename)
    os.makedirs('temp_uploads', exist_ok=True)
    file.save(temp_path)
    # Detect if image or video
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    result_msg = f"Detection completed for {filename}."
    product_counts = {}
    if yolo_model:
        if is_image:
            results = yolo_model(temp_path)
            for r in results:
                image = cv2.imread(temp_path)
                class_indices = r.boxes.cls.int().tolist()
                class_names = r.names
                product_counts = {name: 0 for name in class_names.values()}
                for idx in class_indices:
                    product_counts[class_names[idx]] += 1
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        class_id = box.cls[0].int().item()
                        class_name = r.names[class_id]
                        confidence = box.conf[0].item()
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset = 30
                for name, count in product_counts.items():
                    cv2.putText(image, f"{name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_offset += 30
                _, buffer = cv2.imencode('.png', image)
                img_b64 = base64.b64encode(buffer).decode('utf-8')

                return render_template('result.html', result=result_msg, image_b64=img_b64, product_counts=product_counts)
        elif is_video:
            cap = cv2.VideoCapture(temp_path)  # Open the uploaded video file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join('temp_uploads', 'detected_' + filename)
            out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model(frame)
                for r in results:
                    class_indices = r.boxes.cls.int().tolist()
                    class_names = r.names
                    product_counts = {name: 0 for name in class_names.values()}
                    for idx in class_indices:
                        product_counts[class_names[idx]] += 1
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            class_id = box.cls[0].int().item()
                            class_name = r.names[class_id]
                            confidence = box.conf[0].item()
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset = 30
                    for name, count in product_counts.items():
                        cv2.putText(frame, f"{name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        y_offset += 30
                out.write(frame)
            cap.release()
            out.release()
            video_url = url_for('mediaa', filename='detected_' + filename)

            return render_template('result.html', result=result_msg, video_url=video_url, product_counts=product_counts)
        else:
            result_msg = "Unsupported file type."
    else:
        result_msg = "Model not loaded."
    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass

# Route to serve processed videos from temp_uploads
@app.route('/mediaa/<filename>')
def media(filename):
    return send_from_directory('temp_uploads', filename)

   


# Webcam routes 
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def gen_webcam_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if yolo_model:
            results = yolo_model(frame)
            for r in results:
                class_names = r.names
                counts={name: 0 for name in class_names.values()}
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf[0].item()
                        if confidence>0.30:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            class_id = box.cls[0].int().item()
                            class_name = class_names[class_id]
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # Count the detection
                            counts[class_name] += 1
            # Display counts on screen
            y_offset = 30
            for product, count in counts.items():
                cv2.putText(frame, f"{product}: {count}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            total_count = 0
            for count in counts.values():
                total_count += count
            cv2.putText(frame, f"Total products: {total_count}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    # Release the video capture object
    cap.release()

@app.route('/webcam_feed')
def webcam_feed():
    return Response(gen_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

