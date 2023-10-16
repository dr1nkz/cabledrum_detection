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
        line = line.replace('\n', '').split(', ')
        bbox = [int(l) for l in line]
        bboxes.append(bbox)
    parked_drums_boxes = np.array(bboxes)

    txt = './temp_visual.txt'
    with open(txt) as f:
        lines = f.readlines()
    bboxes = []
    parked_numbers = np.array([])
    for line in lines:
        line = line.replace('\n', '').split(', ')
        parked_numbers = np.append(parked_numbers, line.pop())
        bbox = [int(l) for l in line]
        bbox = np.array(bbox).reshape((-1, 4, 2))
        bboxes.append(bbox)
    parked_drums_boxes_visual = np.array(bboxes)

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
        drums_boxes = np.array(yolov8_detector.get_boxes())

        # чтобы не ломалось iou
        if len(drums_boxes) == 1 or drums_boxes.shape[0] == 1:
            # drums_boxes = np.array([drums_boxes])
            drums_boxes = np.array(drums_boxes).reshape(1, -1)

        # list с координатами пустых мест
        free_space_boxes = []
        free_space_boxes_visual = []
        free_parked_numbers = []
        # free_space = len(parked_drums_boxes) * [False]
        if drums_boxes.shape[0] != 0:
            for i in range(len(parked_drums_boxes)):
                IoUs = compute_iou(parked_drums_boxes[i], drums_boxes)
                max_IoU = np.max(IoUs)
                # Для отладки параметра IoU
                # print(parked_drums_boxes[i])
                # print(max_IoU)
                if max_IoU < 0.35:
                    # Отмечаем, что мы нашли как минимум оно свободное место.
                    # free_space[i] = True
                    free_space_boxes.append(
                        parked_drums_boxes[i].astype('int'))
                    free_space_boxes_visual.append(
                        parked_drums_boxes_visual[i].astype('int'))
                    free_parked_numbers.append(parked_numbers[i])

        else:
            free_space_boxes = parked_drums_boxes.astype('int')
            free_space_boxes_visual = parked_drums_boxes_visual.astype('int')

        # Штриховые рамки вокруг всех парковочных мест.
        for parked_drums_box, parked_drums_box_visual in zip(parked_drums_boxes, parked_drums_boxes_visual):
            x1, y1, x2, y2 = parked_drums_box.astype('int')

            # Штриховая рамка прямоугольника
            # cv2.rectangle(detected_img, (x1, y1), (x2, y2),
            #               (150, 150, 150), thickness=1)

            # Штриховая рамка места под катушкой
            cv2.polylines(detected_img, parked_drums_box_visual,
                          True, (100, 100, 100), thickness=2)

        # Зелёные рамки вокруг пустых мест.
        for free_space_box, free_space_box_visual, free_parked_number in zip(
                free_space_boxes, free_space_boxes_visual, free_parked_numbers):
            x1, y1, x2, y2 = free_space_box.astype('int')

            # Зелёная рамка прямоугольника
            # cv2.rectangle(detected_img, (x1, y1), (x2, y2),
            #               (0, 255, 0), thickness=1)

            # Отображаем надпись Empty place
            fontScale = 1
            thickness = 1
            font = cv2.FONT_HERSHEY_DUPLEX

            # cv2.putText(detected_img, f"Empty place", (x1, y1 - 4 * thickness),
            #             font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

            # Зелёная рамка места под катушкой
            cv2.polylines(detected_img, free_space_box_visual,
                          True, (0, 255, 0), thickness=2)

            x, y = free_space_box_visual[0][0][0], free_space_box_visual[0][0][1]
            # Отображаем номер парковочного места
            cv2.putText(detected_img, free_parked_number, (x, y - 10 * thickness),
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
