from shapely.geometry import Point
from typing import Any, Dict


class Position:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Position):
            self.point = args[0].point
            self.properties = args[0].properties.copy()
        else:
            self.point = Point(*args)
            self.properties = {}
        self.visited = False

    def __hash__(self):
        return id(self)

    def __getattr__(self, item):
        # Delegate attribute access to the internal point
        return getattr(self.point, item)


class Prediction(Position):
    def __init__(self, prediction: Position, truth: Position = None):
        super().__init__(prediction)
        self.truth = truth
