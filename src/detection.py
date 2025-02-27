import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, use_classical_methods: bool = False):
        self._use_classical_methods = use_classical_methods
        self._yolo = YOLO("models/yolo11n-finetuned.pt", task="detect", verbose=False)

    def _run_yolo(self, frame) -> dict:
        results = self._yolo.predict(frame, verbose=False)
        if len(results) == 0:
            return {}

        names = results[0].names
        classes = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)

        objects = {}
        for cls, bbox in zip(classes, boxes):
            x0, y0, x1, y1 = bbox
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            bbox = (x0, y0, w, h)

            name = names[cls]
            if name in objects:
                objects[name].append(bbox)
            else:
                objects[name] = [bbox]

        return objects

    def detect_objects(self, frame: cv2.typing.MatLike) -> dict:
        objects = self._run_yolo(frame)
        if self._use_classical_methods:
            objects.update(detect_petri_dishes(frame))
            objects.update(detect_hands(frame))

        return objects


def detect_objects(frame: cv2.typing.MatLike):
    objects = {}
    objects.update(detect_petri_dishes(frame))
    objects.update(detect_hands(frame))
    return objects


def draw_detections(frame: cv2.typing.MatLike, detections: dict) -> cv2.typing.MatLike:
    for label, instances in detections.items():
        for coords in instances:
            x, y, w, h = coords
            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1
            )
            label_size = cv2.getTextSize(
                label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1
            )
            label_w, label_h = label_size[0]
            # margins
            label_w += 6
            label_h += 6

            frame = cv2.rectangle(
                frame,
                (x, y),
                (x + label_w, y - label_h),
                color=(0, 0, 255),
                thickness=-1,
            )

            cv2.putText(
                frame,
                label,
                (x + 3, y - 3),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=1,
            )

    return frame


def detect_hands(frame: cv2.typing.MatLike) -> dict:
    """Simple method of detecting gloves by colour filtering as the hands are orange"""

    MIN_HAND_SIZE = 20 * 20

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ORANGE_MIN = np.array([5, 50, 180], np.uint8)
    ORANGE_MAX = np.array([15, 255, 255], np.uint8)

    thresholded = cv2.inRange(frame_hsv, ORANGE_MIN, ORANGE_MAX)
    thresholded = cv2.morphologyEx(
        thresholded, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8)
    )
    contours, hierarchy = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected_hands = [cv2.boundingRect(contour) for contour in contours]
    detected_hands = [
        hand for hand in detected_hands if hand[2] * hand[3] > MIN_HAND_SIZE
    ]

    detected_hands.sort(key=lambda hand: hand[2] * hand[3], reverse=True)

    return {"hand": detected_hands[:2]}


def detect_petri_dishes(frame: cv2.typing.MatLike) -> dict:
    """Simple method of detecting petri dishes using Hough Circles"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        frame_gray,
        cv2.HOUGH_GRADIENT,
        2,
        50,
        param1=80,
        param2=100,
        minRadius=60,
        maxRadius=75,
    )

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    empty_dishes = []
    filled_dish = []
    filled_dish_with_lid = []

    if circles is not None and len(circles) > 0:

        for circle in circles[0]:
            x, y, r = np.around(circle).astype(int).tolist()
            # consider a 11 pixel patch about the centre
            roi = frame_hsv[y - 10 : y + 10, x - 10 : x + 10, :]
            hue = np.mean(roi[:, :, 0])
            sat = np.mean(roi[:, :, 1])
            val = np.mean(roi[:, :, 2])

            bbox = (x - r, y - r, r * 2, r * 2)

            if 32 <= hue <= 40 and sat > 15 and val > 40:  # yellowish
                if sat > 35 and val > 120:
                    filled_dish.append(bbox)
                else:
                    filled_dish_with_lid.append(bbox)
            elif sat <= 16 and val > 80:  # basically gray
                empty_dishes.append(bbox)

    return {
        "petri dish filled": filled_dish,
        "petri dish filled with lid": filled_dish_with_lid,
        "petri dish empty": empty_dishes,
    }


if __name__ == "__main__":
    frame = cv2.imread(sys.argv[1])
    start = time.time()
    result = detect_objects(frame)
    end = time.time()

    print(end - start)
    print(result)

    draw_detections(frame, result)
    cv2.imshow("detections", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
