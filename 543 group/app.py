from flask import Flask, request, render_template, redirect, url_for, send_file, send_from_directory
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
from dehaze import Recover, AtmLight, DarkChannel, TransmissionEstimate, TransmissionRefine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')  # Assumes an index.html file with upload form

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file: 
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        mime_type = file.content_type
        
        if 'video' in mime_type:
            output_path = process_video(file_path)
            return f'''
            <!doctype html>
            <html lang="en">
            <head>
                <title>Processed Video</title>
            </head>
            <body>
                <h1>Processed Video</h1>
                <video width="320" height="240" controls>
                  <source src="{url_for('download_file', filename=os.path.basename(output_path))}" type="video/mp4">
                  Your browser does not support the video tag.
                </video>
                <br>
                <a href="/">Upload another file</a>
            </body>
            </html>
            '''
        else:
            image_data_uri = f"data:image/png;base64,{process_image_for_dehaze(file_path)}"
            return f'''
            <!doctype html>
            <html lang="en">
            <head>
                <title>Dehazed Image</title>
            </head>
            <body>
                <h1>Dehazed Image</h1>
                <img src="{image_data_uri}" alt="Dehazed Image">
                <br>
                <a href="/">Upload another image</a>
            </body>
            </html>
            '''
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

def process_image_for_dehaze(file_path):
    with open(file_path, 'rb') as image_file:
        npimg = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Dehazing process
    I = img.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(img, te)
    J = Recover(I, t, A, 0.1)
    
    # Convert the processed image to base64 for embedding in HTML
    _, im_arr = cv2.imencode('.png', J * 255)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode("utf-8")

    return im_b64

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # Determine output path and get properties
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_dehazed.mp4"

    # Using 'X264' or 'avc1' for H.264 encoding
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # Ensure to capture width and height as integers
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    return out_path

def process_frame(frame):
    I = frame.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(frame, te)
    J = Recover(I, t, A, 0.1)
    return (J * 255).astype(np.uint8)

if __name__ == '__main__':
    app.run(debug=True)
