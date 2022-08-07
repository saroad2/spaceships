from pathlib import Path

import click
import pygame

from spaceships.agent import SpaceshipsAgent
from spaceships.cli.spaceships_group import main_cli
from spaceships.env import SpaceshipsEnv


@main_cli.command("auto-play")
@click.option("--size", type=int, default=10)
@click.option("--model-suffix", type=str)
def auto_play_cli(size, model_suffix):
    pygame.init()

    screen = pygame.display.set_mode([500, 500])
    env = SpaceshipsEnv(size, screen=screen)
    agent = SpaceshipsAgent(env=env, batch_size=0, learning_rate=0)
    agent.load_models(Path.cwd() / "models", suffix=model_suffix)
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()

        if not env.lost():
            state = env.state
            action = agent.get_action(state=state, epsilon=0, use_target=True)
            env.step(action)

        env.render()
        pygame.display.flip()
        clock.tick(5)
