#!/usr/bin/env python
from flask import Flask, render_template, Response, request

import cv2

import numpy as np

from PIL import Image

from io import BytesIO

from localStoragePy import localStoragePy

from detector import YOLOv8

from utils import compute_iou

# import onnxruntime as ort

app = Flask(__name__)
localStorage = localStoragePy('test', 'json')


@app.route('/', methods=['GET'])
def index():
    try:
        address = request.args['address']
        localStorage.setItem('address', address)
    except:
        print('incorrect request')
    return render_template('index.html')


def gen(address):
    yield b'--frame\r\n'

    """
    Запуск модели
    """

    # Захват видео с камеры

    # cap = cv2.VideoCapture('./image.jpg')
    cap = cv2.VideoCapture(address)

    model_path = r'./train5/weights/best.onnx'
    yolov8_detector = YOLOv8(path=model_path,
                             conf_thres=0.3,
                             iou_thres=0.5)

    # cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    # Местоположение парковочных мест
    txt = './temp.txt'
    lines = None
    with open(txt) as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(', ')
        bbox = [float(l) for l in line]
        bboxes.append(bbox)
    parked_drums_boxes = np.array(bboxes)

    # Сколько кадров подряд с пустым местом мы уже видели
    free_space_frames = 0

    while cap.isOpened():

        # Кадр с камеры
        ret, frame = cap.read()

        if not ret:
            break

        # Детектирование
        yolov8_detector(frame)
        detected_img = frame
        # detected_img = yolov8_detector.draw_detections(frame)
        drums_boxes = yolov8_detector.get_boxes()

        # чтобы не ломалось iou
        if drums_boxes.shape[0] == 1:
            # drums_boxes = np.array([drums_boxes])
            drums_boxes = np.array(drums_boxes).reshape(1, -1)

        # list с координатами пустых мест
        free_space_boxes = []
        # free_space = len(parked_drums_boxes) * [False]

        for i in range(len(parked_drums_boxes)):
            IoUs = compute_iou(parked_drums_boxes[i], drums_boxes)
            max_IoU = np.max(IoUs)
            if max_IoU < 0.15:
                # Отмечаем, что мы нашли как минимум оно свободное место.
                # free_space[i] = True
                free_space_boxes.append(
                    parked_drums_boxes[i].astype('int'))

        # Штриховые рамки вокруг всех парковочных мест.
        for parked_drums_box in parked_drums_boxes:
            x1, y1, x2, y2 = parked_drums_box.astype('int')
            # print(parked_drums_box)
            # Штриховая рамка

            cv2.rectangle(detected_img, (x1, y1), (x2, y2),
                          (127, 127, 127), thickness=2, lineType=cv2.LINE_8)

        # Зелёные рамки вокруг пустых мест.
        for free_space_box in free_space_boxes:
            x1, y1, x2, y2 = free_space_box

            # Зелёная рамка
            cv2.rectangle(detected_img, (x1, y1), (x2, y2),
                          (0, 255, 0), thickness=2)

            # Отображаем надпись Empty place
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # fontScale
            fontScale = 1

            # Line thickness of 2 px
            thickness = 1

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(detected_img, f"Empty place", (x1, y1 - 4 * thickness),
                        font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

        if detected_img is None:
            continue
        rgb_image = Image.fromarray(cv2.cvtColor(
            detected_img, cv2.COLOR_BGR2RGB), 'RGB')
        buf = BytesIO()
        rgb_image.save(buf, 'JPEG')
        frame = buf.getbuffer()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/video_feed/')
def video_feed():
    address = localStorage.getItem('address')
    if address is None:
        print('no address found')
        return
    return Response(gen(address),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
