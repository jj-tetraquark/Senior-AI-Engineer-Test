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
        self._previous_summary = {}

        self._movement_threshold_px = movement_theshold_px
        self._last_seen_threshold = 3 * 30  # 3 seconds ago
        self._new_object_observation_threshold = 5

        self._known_objects = []

    def update(self, detections, current_time):
        new_objects = self._update_existing_objects_and_determine_new_ones(
            detections, current_time
        )
        self._add_new_objects(new_objects, current_time)
        self._cull_old_objects(current_time)
        summary = self.summarise_known_objects()
        self.update_filled_petri_dish_count(summary)

        self._previous_summary = summary
        return summary


    def update_filled_petri_dish_count(self, summary):
        # TODO make this less jank
        current_filled_dishes = summary.get("petri dish filled", 0)
        previously_filled_dishes = self._previous_summary.get("petri dish filled", 0)
        self.filled_dishes += max(current_filled_dishes - previously_filled_dishes, 0)

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
