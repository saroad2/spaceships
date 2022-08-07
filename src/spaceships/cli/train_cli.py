from pathlib import Path

import click
import numpy as np
import tqdm

from spaceships.agent import SpaceshipsAgent
from spaceships.cli.spaceships_group import main_cli
from spaceships.env import SpaceshipsEnv
from spaceships.history import HistoryPoint
from spaceships.models import update_target
from spaceships.plotting import plot_all

MIN_EPSILON_VALUE = 1e-5


@main_cli.command("train")
@click.option("--size", type=int, default=10)
@click.option("-e", "--episodes", type=int, default=10_000)
@click.option("-b", "--batch-size", type=int, default=64)
@click.option("-w", "--window-size", type=int, default=50)
@click.option("--epsilon", type=float, default=0.5)
@click.option("--epsilon-decay", type=float, default=0.999)
@click.option("-g", "--gamma", type=float, default=0.99)
@click.option("-t", "--tau", type=float, default=0.99)
@click.option("-l", "--learning-rate", type=float, default=0.001)
@click.option("--max-episode-moves", type=int, default=100)
@click.option("--score-stop", type=float, default=100)
@click.option("--checkpoint", type=int, default=1_000)
def train_cli(
    size: int,
    episodes: int,
    batch_size: int,
    window_size: int,
    epsilon: float,
    gamma: float,
    tau: float,
    epsilon_decay: float,
    learning_rate: float,
    max_episode_moves: int,
    score_stop: float,
    checkpoint: int,
):
    env = SpaceshipsEnv(size=size)
    agent = SpaceshipsAgent(env=env, batch_size=batch_size, learning_rate=learning_rate)
    history = []
    plots_dir = Path.cwd() / "plots"
    models_directory = Path.cwd() / "models"
    click.echo(agent.critic.summary())
    max_score = 0
    with tqdm.trange(episodes) as bar:
        for ep in bar:
            agent.run_episode(
                max_episode_moves=max_episode_moves,
                epsilon=epsilon,
            )
            loss = agent.learn(gamma)
            history.append(HistoryPoint.from_env(env=env, loss=loss, epsilon=epsilon))
            update_target(target=agent.target_critic, model=agent.critic, tau=tau)

            latest_history = history[-window_size:]
            means_dict = {
                field: np.mean(
                    [getattr(history_point, field) for history_point in latest_history]
                )
                for field in HistoryPoint.fields()
            }
            scores_mean = means_dict["score"]
            bar.set_description(
                f"Loss: {means_dict['loss']:.2f}, "
                f"Moves: {means_dict['moves'] :.2f}, "
                f"Score: {scores_mean :.2f}, "
                f"Epsilon: {epsilon :.2e}, "
            )
            if scores_mean > 0.6 * max_score:
                epsilon = max(epsilon * epsilon_decay, MIN_EPSILON_VALUE)
            if len(history) > window_size:
                if scores_mean > max_score:
                    agent.save_models(models_directory, suffix="best")
                    max_score = scores_mean
                    bar.set_postfix_str(f"Best: {max_score:.2f}")
                if scores_mean >= score_stop:
                    break
            if (ep + 1) % checkpoint == 0:
                agent.save_models(models_directory)
                plot_all(history=history, window=window_size, output_dir=plots_dir)

    agent.save_models(models_directory)
    plot_all(history=history, window=window_size, output_dir=plots_dir)
