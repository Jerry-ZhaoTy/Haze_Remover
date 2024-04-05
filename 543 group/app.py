from flask import Flask, request, render_template, redirect, url_for
import base64
from dehaze import Recover, AtmLight, DarkChannel, TransmissionEstimate, TransmissionRefine  # Import your dehaze functions here
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def process_image_for_dehaze(image_file):
    # Read the image in OpenCV
    filestr = image_file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Dehaze the image
    I = img.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(img, te)
    J = Recover(I, t, A, 0.1)

    # Convert the processed image to base64 for embedding in HTML
    _, im_arr = cv2.imencode('.png', J * 255)  # im_arr: image in Numpy one-dimension array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode("utf-8")

    return im_b64

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file:
        image_data_uri = f"data:image/png;base64,{process_image_for_dehaze(file)}"
        return f'''
        <!doctype html>
        <html lang="en">
        <head>
            <title>Dehazed Image</title>
        </head>
        <body>
            <h1>Dehazed Image</h1>
            <img src="{image_data_uri}">
            <br>
            <a href="/">Upload another image</a>
        </body>
        </html>
        '''
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
