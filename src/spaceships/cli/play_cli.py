import pygame

from spaceships.cli.spaceships_group import main_cli
from spaceships.direction import Direction
from spaceships.env import SpaceshipsEnv


@main_cli.command("play")
def play_cli():
    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode([500, 500])
    env = SpaceshipsEnv(size=10, screen=screen)

    # Run until the user asks to quit
    running = True
    while running:

        action = None
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = Direction.UP
                if event.key == pygame.K_DOWN:
                    action = Direction.DOWN
                if event.key == pygame.K_LEFT:
                    action = Direction.LEFT
                if event.key == pygame.K_RIGHT:
                    action = Direction.RIGHT
                if event.key == pygame.K_r:
                    env.reset()

        if action is not None and not env.lost():
            env.step(action)
        env.render()

        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()
