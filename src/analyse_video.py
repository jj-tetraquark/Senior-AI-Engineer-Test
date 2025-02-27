import argparse
import time
import yaml

import cv2

from detection import Detector, draw_detections
from event_log import EventLog
from world_state import WorldState

WINDOW_NAME = "Object Detections"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-file", help="Name of video file to analyse", required=True
    )
    parser.add_argument(
        "-o", "--output-log", help="Optional output log file", default=None
    )
    parser.add_argument(
        "--no-display",
        help="Disable video playback with detection annotations",
        action="store_false",
    )

    return parser.parse_args()


def analyse_video(fname, config: dict, output_file: str, display: bool = True):
    world_state = WorldState(**config["world_state"])
    detector = Detector(**config["detector"])

    for obj in config["new_instances_to_track"]:
        world_state.start_tracking_new_instances_of_object_type(obj)

    for obj_a, obj_b, verb, intersection_threshold in config["interactions_to_track"]:
        world_state.start_tracking_object_interactions(
                obj_a, obj_b, intersection_threshold, verb
        )

    cap = cv2.VideoCapture(fname)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    #cap.set(cv2.CAP_PROP_POS_FRAMES, 4250)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rate = int(1000 // fps)
    frame_number = 0

    with EventLog(output_file) as event_log:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            detections = detector.detect_objects(frame)
            state_update = world_state.update(detections, frame_number)
            event_log.update_state(frame_number, state_update)
            end = time.time()

            if display:
                frame = draw_detections(frame, detections)
                cv2.imshow(WINDOW_NAME, frame)
                delay = max(1, int(rate - (end - start) * 1000))

                key = cv2.waitKey(delay)
                if key == ord("q"):
                    break

            frame_number += 1


def main():
    args = parse_args()
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    analyse_video(args.input_file, config, args.output_log, args.no_display)


if __name__ == "__main__":
    main()
