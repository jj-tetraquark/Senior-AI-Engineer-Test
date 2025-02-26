import argparse

import cv2
import time

from detection import Detector, draw_detections
from world_state import WorldState

WINDOW_NAME = "window"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="Name of video file to analyse")

    return parser.parse_args()


def analyse_video(fname, display=True):
    world_state = WorldState()
    detector = Detector()

    world_state.start_tracking_new_instances_of_object_type("petri dish filled")
    world_state.start_tracking_object_interactions(
        "left hand", "petri dish empty", verb="touching"
    )
    world_state.start_tracking_object_interactions(
        "right hand", "petri dish empty", verb="touching"
    )
    world_state.start_tracking_object_interactions(
        "left hand", "petri dish filled", verb="touching"
    )
    world_state.start_tracking_object_interactions(
        "right hand", "petri dish filled", verb="touching"
    )
    world_state.start_tracking_object_interactions(
        "left hand", "petri dish filled with lid", verb="touching"
    )
    world_state.start_tracking_object_interactions(
        "right hand", "petri dish filled with lid", verb="touching"
    )

    world_state.start_tracking_object_interactions(
        "right hand", "bottle", verb="holding"
    )
    world_state.start_tracking_object_interactions(
        "left hand", "bottle", verb="holding"
    )

    world_state.start_tracking_object_interactions(
        "bottle cap", "bottle", verb="attatched to"
    )

    cap = cv2.VideoCapture(fname)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 3250)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rate = int(1000 // fps)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = detector.detect_objects(frame)
        known_objects = world_state.update(detections, frame_number)
        end = time.time()

        if display:
            frame = draw_detections(frame, detections)
            cv2.imshow(WINDOW_NAME, frame)

            delay = max(1, int(rate - (end - start) * 1000))

            key = cv2.waitKey(delay)
            if key == ord("q"):
                break

        frame_number += 1

        print(known_objects)
        print(f"filled dishes: {world_state.get_instance_counts('petri dish filled')}")


def main():
    args = parse_args()
    analyse_video(args.input_file)


if __name__ == "__main__":
    main()
