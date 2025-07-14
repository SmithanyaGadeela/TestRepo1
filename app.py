from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque

# --- CONFIG ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
MODEL_PATH = 'model/fall_3dcnn_model.h5'
SEQUENCE_LENGTH = 64
FRAME_SIZE = 640  # for normalizing keypoints

# --- INIT ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
pose_model = YOLO('yolov8n-pose.pt')  # or path to custom pose model
fall_model = load_model(MODEL_PATH)

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return 'No video uploaded'

    file = request.files['video']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(OUTPUT_FOLDER, f"output_{filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    keypoint_queue = deque(maxlen=SEQUENCE_LENGTH)
    label = "No Fall"
    fall_detected = False  # ⚠️ New flag to lock "Fall" once detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model.predict(frame, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            keypoints_all = results.keypoints.xy

            # Choose largest box
            if len(boxes) > 0:
                areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
                max_idx = np.argmax(areas)
                box = boxes[max_idx]
                x1, y1, x2, y2 = box
                keypoints = keypoints_all[max_idx].cpu().numpy()[:, :2] / FRAME_SIZE
                keypoint_queue.append(keypoints)

                if len(keypoint_queue) == SEQUENCE_LENGTH and not fall_detected:
                    try:
                        pose_seq = np.array(keypoint_queue)
                        pose_seq = np.expand_dims(pose_seq, axis=0)  # [1, 64, 17, 2]
                        prediction = fall_model.predict(pose_seq)[0][0]
                        if prediction > 0.5:
                            label = "Fall"
                            fall_detected = True  # 🔐 Lock once fall is detected
                        else:
                            label = "No Fall"
                    except Exception as e:
                        print("Prediction error:", e)

                # Draw bounding box and label
                color = (0, 0, 255) if label == "Fall" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    return send_from_directory(OUTPUT_FOLDER, f"output_{filename}", as_attachment=True)

# --- MAIN ---
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
