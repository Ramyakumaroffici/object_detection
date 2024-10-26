from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights/yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure output_layers is a NumPy array
output_layers = np.array([layer_names[i - 1] for i in net.getUnconnectedOutLayers()])

def process_frame(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            if detection.shape[0] == 85:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    indices = indices.flatten() if len(indices) > 0 else []

    for idx in indices:
        box = boxes[idx]
        x, y, w, h = map(int, box)  # Ensure x, y, w, h are integers
        print(f"Drawing box with coordinates: x={x}, y={y}, w={w}, h={h}")  # Debug statement
        label = f"{classes[class_ids[idx]]} {confidences[idx]:.2f}"
        color = (0, 255, 0)
        if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):  # Additional check
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print("Invalid coordinates for rectangle:", x, y, w, h)  # Debug statement

    # Save frame as image
    frame_path = os.path.join("static", "frames", "latest_frame.jpg")
    cv2.imwrite(frame_path, frame)  # Overwrites the file each time

    # Encode frame to bytes for further processing if needed
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()
    return frame_data


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/latest_frame")
def latest_frame():
    return send_from_directory("static/frames", "latest_frame.jpg")

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['video']
    video_path = os.path.join("temp_video.mp4")  # Save as a temporary file
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = process_frame(frame)
        frames.append(frame_data)

    cap.release()
    os.remove(video_path)  # Delete the temporary file after processing
    return jsonify({'frames': [f.decode('latin1') for f in frames]})

if __name__ == '__main__':
    # Ensure the frames directory exists
    os.makedirs(os.path.join("static", "frames"), exist_ok=True)
    app.run(debug=True)
