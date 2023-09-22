#!/usr/bin/env python
from flask import Flask, render_template, Response

from detector import onnx_detect_image, onnx_detect_empty_place

from PIL import Image

from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    yield b'--frame\r\n'
    onnx_detect_empty_place()
    while True:
        # img = onnx_detect_image('/home/pc/ML/rtsp/image.jpg')
        img = onnx_detect_image('/app/image.jpg')        
        rgb_image = Image.fromarray(img, 'RGB')
        buf = BytesIO()
        rgb_image.save(buf, 'JPEG')
        frame = buf.getbuffer()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)