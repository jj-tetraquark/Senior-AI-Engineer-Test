from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self):
        return self.w * self.h


@dataclass
class Point:
    x: int
    y: int

    def distance_to(self, point_b: Point) -> float:
        return np.sqrt((self.x - point_b.x) ** 2 + (self.y - point_b.y) ** 2)


def intersection_area_of_two_bboxs(bbox1: BoundingBox, bbox2: BoundingBox) -> int:
    x_left = max(bbox1.x, bbox2.x)
    y_top = max(bbox1.y, bbox2.y)
    x_right = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
    y_bottom = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)

    x_overlap = max(0, x_right - x_left)
    y_overlap = max(0, y_bottom - y_top)

    return x_overlap * y_overlap
