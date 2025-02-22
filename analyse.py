import argparse
import cv2

from detection import detect_objects, draw_detections

WINDOW_NAME = "window"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="Name of video file to analyse")

    return parser.parse_args()

def analyse_video(fname, display=True):
    cap = cv2.VideoCapture(fname)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1650)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rate = int(1000//fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        if display:
            frame = draw_detections(frame, detections)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


def main():
    args = parse_args()
    analyse_video(args.input_file)


if __name__ == '__main__':
    main()
