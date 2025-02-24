import itertools
from dataclasses import dataclass

import numpy as np

from utils import BoundingBox, Point, intersection_area_of_two_bboxs


@dataclass
class InstanceObject:
    object_type: str
    bbox: BoundingBox
    observation_count: int = 0
    last_seen: int = -1

    @property
    def centroid(self):
        return Point(self.bbox.x + self.bbox.w / 2, self.bbox.y + self.bbox.h / 2)


class WorldState:
    def __init__(self, movement_theshold_px=30):
        self.filled_dishes = 0

        self._movement_threshold_px = movement_theshold_px
        self._last_seen_threshold = 3 * 30  # 3 seconds ago
        self._new_object_observation_threshold = 5

        self._known_objects = []
        self._tracked_new_object_counts = {}
        self._interactions_to_track = []

    def update(self, detections: dict, current_time: int):
        new_objects = self._update_existing_objects_and_identify_new_ones(
            detections, current_time
        )
        self._add_new_objects(new_objects, current_time)
        self._cull_old_objects(current_time)

        self.update_new_instance_counts()
        self.detect_interactions()
        return self.summarise_known_objects()

    def start_tracking_new_instances_of_object_type(self, object_type: str):
        """
        Initialises a counter to track the amount of times a new object of a given type
        appears in the scene.
        """
        self._tracked_new_object_counts[object_type] = 0

    def update_new_instance_counts(self):
        for obj in self._known_objects:
            if (
                obj.object_type in self._tracked_new_object_counts
                and obj.observation_count == self._new_object_observation_threshold
            ):
                self._tracked_new_object_counts[obj.object_type] += 1

    def get_instance_counts(self, object_type):
        return self._tracked_new_object_counts[object_type]

    def start_tracking_object_interactions(
        self,
        object_a: str,
        object_b: str,
        intersection_threshold=0.5,
        verb: str = "interacting with",
    ):
        self._interactions_to_track.append(
            (object_a, object_b, intersection_threshold, verb)
        )

    def detect_interactions(self):
        for interaction in self._interactions_to_track:
            object_a, object_b, intersection_thresh, verb = interaction
            object_a_bboxes = [
                obj.bbox for obj in self._known_objects if obj.object_type == object_a
            ]
            object_b_bboxes = [
                obj.bbox for obj in self._known_objects if obj.object_type == object_b
            ]

            for a_bbox, b_bbox in itertools.product(object_a_bboxes, object_b_bboxes):
                intersection_area = intersection_area_of_two_bboxs(a_bbox, b_bbox)
                intersection_thresh_px = intersection_thresh * min(
                    a_bbox.area, b_bbox.area
                )

                if intersection_area >= intersection_thresh_px:
                    print(f"{object_a} is {verb} {object_b}")

    def summarise_known_objects(self):
        object_counts = {}
        for obj in self._known_objects:
            if obj.observation_count < self._new_object_observation_threshold:
                continue

            if obj.object_type in object_counts:
                object_counts[obj.object_type] += 1
            else:
                object_counts[obj.object_type] = 1

        return object_counts

    def _update_existing_objects_and_identify_new_ones(self, detections, current_time):
        new_objects = []
        for object_type, bounding_boxes in detections.items():
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                x += w / 2
                y += h / 2
                already_known = False
                for known_object in self._known_objects:
                    if object_type == known_object.object_type:
                        distance = known_object.centroid.distance_to(Point(x, y))
                        if distance < self._movement_threshold_px:
                            known_object.bbox = BoundingBox(*bbox)
                            known_object.observation_count += 1
                            known_object.last_seen = current_time
                            already_known = True
                            break

                if not already_known:
                    new_objects.append((object_type, bbox))

        return new_objects

    def _add_new_objects(self, new_objects, current_time):
        for object_type, bbox in new_objects:
            self._known_objects.append(
                InstanceObject(
                    object_type,
                    BoundingBox(*bbox),
                    observation_count=1,
                    last_seen=current_time,
                )
            )

    def _cull_old_objects(self, current_time):
        self._known_objects = [
            obj
            for obj in self._known_objects
            if not self._should_cull_object(obj, current_time)
        ]

    def _should_cull_object(self, obj, current_time):
        return obj.last_seen < (current_time - self._last_seen_threshold) or (
            obj.observation_count < self._new_object_observation_threshold
            and obj.last_seen < (current_time - 10)
        )
