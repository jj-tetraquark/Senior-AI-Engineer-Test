import numpy as np
from dataclasses import dataclass


@dataclass
class InstanceObject:
    object_type: str
    x: int = -1
    y: int = -1
    observation_count: int = 0
    last_seen: int = -1


class WorldState:
    def __init__(self, movement_theshold_px=30):
        self.filled_dishes = 0

        self._movement_threshold_px = movement_theshold_px
        self._last_seen_threshold = 3 * 30  # 3 seconds ago
        self._new_object_observation_threshold = 5

        self._known_objects = []
        self._tracked_new_object_counts = {}

    def update(self, detections: dict, current_time: int):
        new_objects = self._update_existing_objects_and_determine_new_ones(
            detections, current_time
        )
        self._add_new_objects(new_objects, current_time)
        self._cull_old_objects(current_time)

        self.update_new_instance_counts()

    def start_tracking_new_instances_of_object_type(self, object_type: str):
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

    def _update_existing_objects_and_determine_new_ones(self, detections, current_time):
        new_objects = []
        for object_type, instances in detections.items():
            for instance in instances:
                x, y, w, h = instance
                x += w / 2
                y += h / 2
                already_known = False
                for known_object in self._known_objects:
                    if object_type == known_object.object_type:
                        # distance to centroid
                        distance = np.sqrt(
                            (known_object.x - x) ** 2 + (known_object.y - y) ** 2
                        )
                        # print(f"{object_type}: {distance}")
                        if distance < self._movement_threshold_px:
                            known_object.x = x
                            known_object.y = y
                            known_object.observation_count += 1
                            known_object.last_seen = current_time
                            already_known = True
                            break

                if not already_known:
                    new_objects.append((object_type, instance))

        return new_objects

    def _add_new_objects(self, new_objects, current_time):
        for object_type, (x, y, w, h) in new_objects:
            x = x + w / 2
            y = y + h / 2
            self._known_objects.append(
                InstanceObject(
                    object_type,
                    x,
                    y,
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
