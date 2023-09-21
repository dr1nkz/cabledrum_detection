import time
import cv2
import numpy as np
import onnxruntime


from utils import xywh2xyxy, nms, compute_iou


class YOLOv8:
    """
    Модель YOLO, преобразованная в onnx формат
    """

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Инициализация модели
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        """
        Инициализация модели

        :param path: путь к модели
        """
        # Основной класс для запуска модели
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Получение информации о модели
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        """
        Детекция изображения

        :param image: np.array - прочитанное изображение в массив
        """
        input_tensor = self.prepare_input(image)

        # Результат предикции
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        """
        Подготавливает изображение

        :param image: np.array - прочитанное изображение в массив
        """
        self.img_height, self.img_width = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ресайз изображения
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Скалирование изображения
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        input_tensor = image[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        """
        Инференс модели

        :param input_tensor: np.array - подготовленное изображение
        """
        start = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        """
        Подготовка результатов модели
        """
        predictions = np.squeeze(output[0]).T

        # Фильтрафия оценок, которые ниже уверенности модели
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Класс с наибольшей уверенностью
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Прямоугольники для каждого предсказания
        self.extract_boxes(predictions)

        # Применение метода nms
        indices = nms(self.boxes, scores, self.iou_threshold)

        return self.boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        """
        Извлечение прямоугольников
        """
        # Прямоугольники
        self.boxes = predictions[:, :4]
        # Рескалинг под разрешение изображения
        self.boxes = self.rescale_boxes(self.boxes)
        # Перевод в формат vol
        self.boxes = xywh2xyxy(self.boxes)

    def get_boxes(self):
        """
        Получить прямоугольники из экземпляра класса
        """
        return self.boxes

    def rescale_boxes(self, boxes):
        """
        Рескейл к исходному разрешению
        """
        input_shape = np.array([self.input_width, self.input_height,
                                self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height,
                          self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image):
        """
        Нанесение прямоугольников
        """
        class_names = ['Cable_drum']
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(len(class_names), 3))

        # Прямоугольники
        for box, score, class_id in zip(self.boxes, self.scores, self.class_ids):
            color = colors[class_id]

            x_1, y_1, x_2, y_2 = box.astype(int)

            # Прямоугольник
            cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color, 2)

            # Отображение лейблов
            label = class_names[class_id]
            caption = f'{label} {int(score * 100)}%'

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # fontScale
            fontScale = 1

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            cv2.putText(image, caption, (x_1, y_1 - 4 * thickness),
                        font, fontScale, color, thickness, cv2.LINE_AA)

        return image

    def get_input_details(self):
        """
        Получение информации из входных данных
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        """
        Информация о выходных значениях
        """
        model_outputs = self.session.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))]


def onnx_detect_image(image_path):
    """
    Запуск модели
    """
    # model_path = r'train4\weights\best.onnx'
    model_path = r'./train5/weights/best.onnx'

    yolov8_detector = YOLOv8(path=model_path,
                             conf_thres=0.3,
                             iou_thres=0.5)

    img = cv2.imread(image_path)

    yolov8_detector(img)
    detected_img = yolov8_detector.draw_detections(img)

    # plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def onnx_detect_video():
    """
    Запуск модели
    """
    # Захват видео с камеры
    cap = cv2.VideoCapture(0)

    # model_path = r'train4\weights\best.onnx'
    model_path = r'./train5/weights/best.onnx'

    yolov8_detector = YOLOv8(path=model_path,
                             conf_thres=0.3,
                             iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    while cap.isOpened():

        # Кадр с камеры
        ret, frame = cap.read()

        if not ret:
            break

        # Детектирование
        yolov8_detector(frame)
        detected_img = yolov8_detector.draw_detections(frame)

        # plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))

        cv2.imshow("Detected Objects", detected_img)

        # Для выхода нажать q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def onnx_detect_empty_place():
    """
    Запуск модели
    """
    print('Type the address of rtsp camera:')
    address = input()
    # Захват видео с камеры
    cap = cv2.VideoCapture(address)

    model_path = r'./train5/weights/best.onnx'
    yolov8_detector = YOLOv8(path=model_path,
                             conf_thres=0.3,
                             iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    # Местоположение парковочных мест
    parked_drums_boxes = None
    # Сколько кадров подряд с пустым местом мы уже видели
    free_space_frames = 0
    # Пустые места
    free_space_boxes = None

    while cap.isOpened():

        # Кадр с камеры
        ret, frame = cap.read()

        if not ret:
            break

        # Детектирование
        yolov8_detector(frame)
        detected_img = yolov8_detector.draw_detections(frame)

        if parked_drums_boxes is None:
            parked_drums_boxes = yolov8_detector.get_boxes()
        else:
            # Если новая катушка >
            # Если катушка только попала в кадр =
            if len(yolov8_detector.get_boxes()) >= len(parked_drums_boxes):
                parked_drums_boxes = yolov8_detector.get_boxes()

            all_spaces_free = False

            drums_boxes = yolov8_detector.get_boxes()
            # Если ни одна катушка не найдена, то только нарисовать пустые места
            if (drums_boxes is None) or (len(drums_boxes) == 0):
                if free_space_boxes is None:
                    continue
                else:
                    all_spaces_free = True
                    # free_space_boxes = parked_drums_boxes

            if not (all_spaces_free):
                # Если только 1 катушка, то чтобы не ломалось IoU
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

            # Зелёные рамки вокруг пустых мест.
            for free_space_box in free_space_boxes:
                x1, y1, x2, y2 = free_space_box

                # Зелёная рамка
                cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Отображаем надпись Empty place
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # fontScale
                fontScale = 1

                # Line thickness of 2 px
                thickness = 2

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(detected_img, f"Empty place", (x1, y1 - 4 * thickness),
                            font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

        cv2.imshow("Detected Objects", detected_img)

        # Для выхода нажать q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap = None
            cv2.destroyAllWindows()
            break

    cap = None
    cv2.destroyAllWindows()


print('Type the path for image:')
path = input()
onnx_detect_image(path)
# onnx_detect_empty_place()
