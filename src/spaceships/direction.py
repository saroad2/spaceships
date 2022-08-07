from enum import IntEnum

import numpy as np


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def to_vector(self):
        return _DIRECTION_TO_VECTOR[self]

    def to_one_hot(self):
        one_hot = np.zeros(shape=len(Direction))
        one_hot[self.value] = 1
        return one_hot


_DIRECTION_TO_VECTOR = {
    Direction.UP: np.array([0, -1]),
    Direction.DOWN: np.array([0, 1]),
    Direction.LEFT: np.array([-1, 0]),
    Direction.RIGHT: np.array([1, 0]),
}
