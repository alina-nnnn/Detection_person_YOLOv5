import torch
import cv2
import argparse

def draw_detections(frame, detections, model_names, confidence_threshold=0.5):
    """
    Делает рисовку детектируемых объектов (людей) и выводит уверенность (точность) в распознании.

    Параметры:
    ----------
    frame : np.ndarray
      Кадр видео, на котором выполняется детекция
    detections : numpy.ndarray
      Массив детекций с координатами боксов, уверенностью и классом
    model_names : list
      Список имен классов модели (person - 0)
    confidence_threshold : float
      Порог уверенности для отображения детекции

    Возвращаемое значение:
    ----------------------
    frame : numpy.ndarray
      Кадр с отрисованными детекциями
    """

    for *box, conf, cls in detections:
        if int(cls) == 0 and conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f'{model_names[int(cls)]} {conf:.2f}'
            text = f'{label} ({conf * 100:.1f}%)'
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def main(args):
    """
    Обработка видео с использованием модели YOLOv5.

    Параметры:
    ----------
    args
        Аргументы командной строки (--input и --output)
    """
    # Загрузка модели YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

    # Открытие видеофайла
    input_video = cv2.VideoCapture(args.input)

    # Получение параметров видео
    frame_width = int(input_video.get(3))
    frame_height = int(input_video.get(4))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))

    # Создание объекта для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break

        # Преобразование изображения и выполнение детекции
        results = model(frame)

        # Получение детекций
        detections = results.xyxy[0].cpu().numpy()

        # Отрисовка объектов и точности
        frame = draw_detections(frame, detections, model.names)

        # Запись кадра в выходное видео
        out_video.write(frame)

    # Освобождение ресурсов
    input_video.release()
    out_video.release()

    print(f"Видео с детекцией людей сохранен в: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Детекция людей на видео с использованием YOLOv5')
    parser.add_argument('--input', required=True, type=str, help='Путь к первоначальному видео')
    parser.add_argument('--output', required=True, type=str, help='Путь для сохранения видео с детекцией')
    args = parser.parse_args()

    main(args)

