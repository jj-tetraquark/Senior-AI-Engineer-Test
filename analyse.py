import argparse
import cv2

from detection import detect_objects, draw_detections
from state_manager import WorldState

WINDOW_NAME = "window"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="Name of video file to analyse")

    return parser.parse_args()

def analyse_video(fname, display=True):
    world_state = WorldState()

    cap = cv2.VideoCapture(fname)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    #cap.set(cv2.CAP_PROP_POS_FRAMES, 3250)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rate = int(1000//fps)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)
        known_objects = world_state.update(detections, frame_number)

        if display:
            frame = draw_detections(frame, detections)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        frame_number += 1

        print(known_objects)
        print(f"filled dishes: {world_state.filled_dishes}")


def main():
    args = parse_args()
    analyse_video(args.input_file)

if __name__ == '__main__':
    main()
