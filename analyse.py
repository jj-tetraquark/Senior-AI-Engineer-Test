import argparse

import cv2

from detection import Detector, draw_detections
from state_manager import WorldState

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

        detections = detector.detect_objects(frame)
        known_objects = world_state.update(detections, frame_number)

        if display:
            frame = draw_detections(frame, detections)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(0)
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
