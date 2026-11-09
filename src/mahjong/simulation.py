"""Run reproducible random-policy MahJax episodes from the command line."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Sequence

from .agents import RandomAgent
from .environment import MahJaxEnvironment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics directly exposed by one completed MahJax episode."""

    returns: tuple[float, ...]
    steps: int
    duration_seconds: float
    final_scores: tuple[int, ...] | None
    final_ranking: tuple[int, ...] | None


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregate metrics from a sequential simulation run."""

    episodes: tuple[EpisodeResult, ...]
    duration_seconds: float

    @property
    def average_player_zero_return(self) -> float:
        """Mean undiscounted return for player 0 across the completed episodes."""
        return sum(result.returns[0] for result in self.episodes) / len(self.episodes)

    @property
    def throughput_steps_per_second(self) -> float:
        """Completed environment steps divided by elapsed wall-clock time."""
        steps = sum(result.steps for result in self.episodes)
        return steps / self.duration_seconds if self.duration_seconds else 0.0


def run_simulation(
    episodes: int, *, seed: int = 0, environment_id: str = "no_red_mahjong"
) -> SimulationSummary:
    """Run complete single-hand episodes with four independent random policies.

    A fresh environment and deterministically derived policy seeds are used for
    each episode. MahJax's PRNG key controls game randomness; Python's random
    generator controls only policy choices.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    started = time.perf_counter()
    results: list[EpisodeResult] = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        environment = MahJaxEnvironment(
            environment_id=environment_id, round_mode="single", seed=episode_seed
        )
        agents = [RandomAgent(seed=episode_seed * 4 + player) for player in range(4)]
        transition = environment.reset()
        returns = [0.0] * 4
        steps = 0
        episode_started = time.perf_counter()

        while not transition.done:
            player = transition.current_player
            action = agents[player].select_action(
                transition.observation, transition.legal_actions
            )
            transition = environment.step(action)
            returns = [total + reward for total, reward in zip(returns, transition.rewards)]
            steps += 1

        results.append(
            EpisodeResult(
                returns=tuple(returns),
                steps=steps,
                duration_seconds=time.perf_counter() - episode_started,
                final_scores=environment.final_scores(),
                final_ranking=environment.final_ranking(),
            )
        )
        LOGGER.debug("Completed episode %s in %s steps.", episode_index, steps)

    return SimulationSummary(tuple(results), time.perf_counter() - started)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random-agent MahJax single-hand episodes.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of complete episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic base seed.")
    parser.add_argument(
        "--environment",
        choices=("no_red_mahjong", "red_mahjong"),
        default="no_red_mahjong",
        help="MahJax ruleset; the default is the documented simplified baseline.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and print only metrics exposed by the environment."""
    arguments = _parser().parse_args(argv)
    summary = run_simulation(
        arguments.episodes, seed=arguments.seed, environment_id=arguments.environment
    )
    print(f"Episodes: {len(summary.episodes)}")
    print(f"Average player-0 return: {summary.average_player_zero_return:.3f}")
    print(f"Execution time: {summary.duration_seconds:.3f} s")
    print(f"Simulation throughput: {summary.throughput_steps_per_second:.1f} steps/s")
    last = summary.episodes[-1]
    print(f"Last episode duration: {last.duration_seconds:.3f} s")
    print(f"Last final ranking (player IDs): {last.final_ranking}")
    print(f"Last final scores (hundreds of points): {last.final_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
