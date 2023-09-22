import numpy as np


def nms(boxes, scores, iou_threshold):
    """
    Алгоритм nms для удаления дублирующихся рамок
    """
    # Сортировка по значению предсказания
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Выбор последнего прямоугольника
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Вычисление метрики по сравнению с остальными
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Выбор боксов, у которых метрика не превышает порога
        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    """
    Вычисление iou
    """
    # Выбор минимальных/максимальных значений
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Вычисление площади пересечения
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Площадь объединения
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Расчет IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    """
    Конвертация формата рамок из YOLO в VOC
    """
    # Конвертация (x, y, w, h) в (x1, y1, x2, y2)
    # Из yolo формата в VOC
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
