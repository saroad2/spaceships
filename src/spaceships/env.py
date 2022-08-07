from typing import List, Optional, Tuple, Union

import gym
import numpy as np
import pygame
from gym.core import RenderFrame

from spaceships.colors import BLACK, BLUE, WHITE, YELLOW
from spaceships.direction import Direction


class SpaceshipsEnv(gym.Env):
    star_reward = 10
    move_reward = 1
    lost_penalty = 2

    def __init__(self, size: int, screen: Optional[pygame.Surface] = None):
        self.size = size
        self.screen = screen
        self.font = (
            pygame.font.SysFont("ariel", 24) if self.screen is not None else None
        )
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(self.size, self.size), dtype=int
        )
        self.action_space = gym.spaces.Discrete(4)
        self.player_location = self.random_location()
        self.star_location = self.random_location()
        self.moves = 0
        self.score = 0
        self.star_hits = 0

    @property
    def state_shape(self):
        return self.size, self.size, 2

    @property
    def state(self):
        state = np.zeros(shape=self.state_shape)
        if not self.lost():
            px, py = self.player_location
            state[px, py, 0] = 1
        sx, sy = self.star_location
        state[sx, sy, 1] = 1
        return state

    @property
    def distance_to_star(self):
        return np.linalg.norm(self.player_location - self.star_location)

    @property
    def screen_width(self):
        if self.screen is None:
            return 0
        return self.screen.get_width()

    @property
    def block_size(self):
        return self.screen_width // self.size

    def lost(self):
        if self.player_location[0] < 0 or self.player_location[0] >= self.size:
            return True
        if self.player_location[1] < 0 or self.player_location[1] >= self.size:
            return True
        return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        self.player_location = self.random_location()
        self.star_location = self.random_location()
        self.moves = 0
        self.score = 0
        self.star_hits = 0
        return self.state

    def step(self, action: Direction) -> Tuple[np.ndarray, float, bool, dict]:
        self.moves += 1
        self.player_location += action.to_vector()
        lost = self.lost()
        if lost:
            reward = -self.lost_penalty
        elif np.array_equal(self.player_location, self.star_location):
            self.star_hits += 1
            reward = self.star_reward
            self.star_location = self.random_location()
        else:
            reward = self.move_reward / self.distance_to_star
        self.score += reward
        return self.state, reward, lost, {}

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.screen is None:
            return None
        self.screen.fill(WHITE)
        self.draw_block(location=self.player_location, color=BLUE)
        self.draw_block(location=self.star_location, color=YELLOW)
        img = self.font.render(f"Score: {self.score:.2f}", False, BLACK)
        rect = img.get_rect()
        rect.midtop = (self.screen_width // 2, 0)
        self.screen.blit(img, rect)
        return None

    def random_location(self):
        return np.random.randint(self.size, size=2, dtype=int)

    def draw_block(self, location, color):
        x, y = location * self.block_size
        pygame.draw.rect(
            self.screen, color=color, rect=(x, y, self.block_size, self.block_size)
        )