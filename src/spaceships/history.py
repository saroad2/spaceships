from dataclasses import dataclass, fields

from spaceships.env import SpaceshipsEnv


@dataclass
class HistoryPoint:
    moves: int
    star_hits: int
    score: float
    loss: float
    epsilon: float

    @classmethod
    def fields(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def from_env(cls, env: SpaceshipsEnv, loss: float, epsilon: float):
        return HistoryPoint(
            moves=env.moves,
            star_hits=env.star_hits,
            score=env.score,
            loss=loss,
            epsilon=epsilon,
        )
