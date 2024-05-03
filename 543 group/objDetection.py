# objDetection.py
# Since weights file is larger than 100MB, please download the file through the link and put it under the file '543 group'
# (path same as other py files). https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
import cv2
import os
import numpy as np
import base64
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
# print(net.getUnconnectedOutLayers())

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objs(file_path):
    img = cv2.imread(file_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]*100:.2f}%"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, im_arr = cv2.imencode('.png', img)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode("utf-8")
    return f"data:image/png;base64,{im_b64}"


def upload_and_detect(app):
    @app.route('/upload_to_eval', methods=['POST'])
    def upload_to_eval():
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_data_uri = detect_objs(file_path)
            return f'''
            <!doctype html>
            <html lang="en">
            <head>
                <title>Detected Objects</title>
            </head>
            <body>
                <h1>Detected Objects</h1>
                <img src="{image_data_uri}" alt="Detected Objects">
                <br>
                <a href="/">Upload another image</a>
            </body>
            </html>
            '''
        return redirect(url_for('index'))
