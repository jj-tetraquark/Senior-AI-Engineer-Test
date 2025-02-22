import cv2
import time
import numpy as np


def detect_objects(frame):
    objects = {}
    objects.update(detect_petri_dishes(frame))
    objects.update(detect_hands(frame))
    return objects


def draw_detections(frame, detections):
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


def detect_hands(frame):
    """Simple method of detecting gloves by colour filtering as the hands are orange"""
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

    drawn = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 3)

    detected_hands = [cv2.boundingRect(contour) for contour in contours]
    detected_hands.sort(key=lambda hand: hand[2] * hand[3], reverse=True)

    return {"hand": detected_hands[:2]}


def detect_petri_dishes(frame):
    """Simple method of detecting petri dishes using Hough Circles"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        frame_gray,
        cv2.HOUGH_GRADIENT,
        2,
        50,
        param1=100,
        param2=100,
        minRadius=67,
        maxRadius=75,
    )

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    empty_dishes = []
    full_dishes = []

    if len(circles) == 0:
        return

    for circle in circles[0]:
        x, y, r = np.around(circle).astype(int).tolist()
        # consider a 11 pixel patch about the centre
        roi = frame_hsv[y - 10 : y + 10, x - 10 : x + 10, :]
        hue = np.mean(roi[:, :, 0])
        sat = np.mean(roi[:, :, 1])
        val = np.mean(roi[:, :, 2])

        if 32 <= hue <= 40 and sat > 15 and val > 40:  # yellowish
            full_dishes.append((x - r, y - r, r * 2, r * 2))
        elif sat <= 16 and val > 80:  # basically gray
            empty_dishes.append((x - r, y - r, r * 2, r * 2))

        test_frame = frame.copy()
        cv2.circle(test_frame, (x, y), r, (0, 255, 0), 2)
        #print(f"hue: {hue}")
        #print(f"sat: {sat}")
        #print(f"val: {val}")
        #cv2.imshow("circle", test_frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    return {"petri dish full": full_dishes, "petri dish empty": empty_dishes}


if __name__ == "__main__":
    frame = cv2.imread("test_frame.png")
    start = time.time()
    result = detect_objects(frame)
    end = time.time()

    print(end - start)
    print(result)

    draw_detections(frame, result)
    cv2.imshow("detections", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
