from typing import List, Optional, Tuple, Union

import gym
import numpy as np
import pygame
from gym.core import RenderFrame

from spaceships.colors import BLACK, BLUE, GREEN, RED, WHITE, YELLOW
from spaceships.direction import Direction


class SpaceshipsEnv(gym.Env):
    star_reward = 10
    enemy_death_reward = 5
    move_reward = 1
    lost_penalty = 2

    def __init__(
        self,
        size: int,
        enemy_move_chance: float = 0.33,
        turn_chance: float = 0,
        holes_number: int = 10,
        screen: Optional[pygame.Surface] = None,
    ):
        self.size = size
        self.enemy_move_chance = enemy_move_chance
        self.turn_chance = turn_chance
        self.holes_number = holes_number
        self.screen = screen
        self.font = (
            pygame.font.SysFont("ariel", 24) if self.screen is not None else None
        )
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(self.size, self.size), dtype=int
        )
        self.action_space = gym.spaces.Discrete(4)
        self.holes = self.generate_holes()
        self.player_location = self.generate_player()
        self.enemy_location = self.generate_npc()
        self.star_location = self.generate_npc()
        self.moves = 0
        self.score = 0
        self.enemy_kills = 0
        self.star_hits = 0
        self.slided = False

    @property
    def state_shape(self):
        return self.size, self.size, 4

    @property
    def state(self):
        state = np.zeros(shape=self.state_shape)
        if self.lost():
            return state
        px, py = self.player_location
        state[px, py, 0] = 1
        sx, sy = self.star_location
        state[sx, sy, 1] = 1
        ex, ey = self.enemy_location
        state[ex, ey, 2] = 1
        for hole in self.holes:
            hx, hy = hole
            state[hx, hy, 3] = 1
        return state

    @property
    def max_distance(self):
        return np.sqrt(2) * self.size

    @property
    def distance_to_star(self):
        return np.linalg.norm(self.player_location - self.star_location)

    @property
    def distance_to_enemy(self):
        return np.linalg.norm(self.player_location - self.enemy_location)

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
        if np.array_equal(self.player_location, self.enemy_location):
            return True
        if self.in_hole(self.player_location):
            return True
        return False

    def in_hole(self, location):
        return np.any([np.array_equal(location, hole) for hole in self.holes])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        self.holes = self.generate_holes()
        self.player_location = self.generate_player()
        self.enemy_location = self.generate_npc()
        self.star_location = self.generate_npc()
        self.moves = 0
        self.score = 0
        self.enemy_kills = 0
        self.star_hits = 0
        self.slided = False
        return self.state

    def step(self, action: Direction) -> Tuple[np.ndarray, float, bool, dict]:
        self.moves += 1
        self.move_player(action)
        self.move_enemy()
        lost = self.lost()
        if lost:
            reward = -self.lost_penalty
        elif np.array_equal(self.player_location, self.star_location):
            self.star_hits += 1
            reward = self.star_reward
            self.star_location = self.generate_npc()
        elif self.in_hole(self.enemy_location):
            self.enemy_kills += 1
            self.enemy_location = self.generate_npc()
            reward = self.enemy_death_reward
        else:
            reward = (
                self.move_reward
                * (
                    1 / self.distance_to_star
                    + self.distance_to_enemy / self.max_distance
                )
                / 2
            )
        self.score += reward
        return self.state, reward, lost, {}

    def move_player(self, action):
        turn_index = np.random.choice(
            [-1, 0, 1],
            p=[self.turn_chance / 2, 1 - self.turn_chance, self.turn_chance / 2],
        )
        direction_index = np.mod(action.value + turn_index, len(Direction))
        self.player_location += Direction(direction_index).to_vector()
        self.slided = turn_index != 0

    def move_enemy(self):
        if np.array_equal(self.player_location, self.enemy_location):
            return
        if np.random.uniform() > self.enemy_move_chance:
            return
        move_vector = self.player_location - self.enemy_location
        left_right_vector = Direction.RIGHT if move_vector[0] > 0 else Direction.LEFT
        up_down_vector = Direction.DOWN if move_vector[1] > 0 else Direction.UP
        direction_vector = (
            left_right_vector
            if np.abs(move_vector[0]) > np.abs(move_vector[1])
            else up_down_vector
        )
        self.enemy_location += direction_vector.to_vector()

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.screen is None:
            return None
        self.screen.fill(WHITE)
        player_color = GREEN if self.slided else BLUE
        self.draw_block(location=self.star_location, color=YELLOW)
        self.draw_block(location=self.player_location, color=player_color)
        self.draw_block(location=self.enemy_location, color=RED)
        for hole in self.holes:
            self.draw_block(location=hole, color=BLACK)
        img = self.font.render(f"Score: {self.score:.2f}", False, BLACK)
        rect = img.get_rect()
        rect.midtop = (self.screen_width // 2, 0)
        self.screen.blit(img, rect)
        return None

    def generate_holes(self):
        return [self.random_location() for _ in range(self.holes_number)]

    def generate_player(self):
        location = self.random_location()
        while self.in_hole(location):
            location = self.random_location()
        return location

    def generate_npc(self):
        location = self.random_location()
        while self.in_hole(location) or np.array_equal(location, self.player_location):
            location = self.random_location()
        return location

    def random_location(self):
        return np.random.randint(self.size, size=2, dtype=int)

    def draw_block(self, location, color):
        x, y = location * self.block_size
        pygame.draw.rect(
            self.screen, color=color, rect=(x, y, self.block_size, self.block_size)
        )
