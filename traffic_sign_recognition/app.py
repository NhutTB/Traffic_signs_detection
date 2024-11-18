
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['JSON_AS_ASCII'] = False

model = YOLO(r"D:\Document\FALL_2024\DPL302m\Yolov8_main\traffic_sign_TDT.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Không tìm thấy file', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'Chưa chọn file', 400
    
    if not allowed_file(file.filename):
        return 'Loại file không được hỗ trợ', 400
    
    try:
        # Lưu file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        results = model(file_path)
        detections = []
        img = cv2.imread(file_path)
        img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = r.names[class_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({confidence:.2%})"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

                cv2.putText(img, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 2)
                })

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
        cv2.imwrite(output_path, img)
        
        return render_template('result.html', 
                             detected_image='detected_' + filename,
                             detections=detections)
    
    except Exception as e:
        return f'Lỗi xử lý: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)