"""Evaluate full no-red MahJax hanchan without changing the single-hand baseline."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .agents import RandomAgent
from .environment import MahJaxEnvironment


class Policy(Protocol):
    """Policy input is limited to the current player's observation and legal IDs."""

    def select_action(self, observation: Any, legal_actions: Sequence[int]) -> int: ...


@dataclass(frozen=True)
class KyokuOutcome:
    """One completed hand, irrespective of the round number or dealer repeats.

    The last hand's raw score delta includes any match-end settlement made
    by MahJax, so completed-hand deltas sum to the completed match delta.
    """

    round_index: int
    honba: int
    focal_score_delta: int
    ron_source: int | None
    chankan: bool
    focal_deal_in: bool


@dataclass(frozen=True)
class HanchanResult:
    """Separate simulator reward, score change and research penalties."""

    focal_player: int
    steps: int
    kyoku_outcomes: tuple[KyokuOutcome, ...]
    raw_return: float
    score_delta: int
    deal_in_penalty: float
    placing_penalty: float
    final_place: int | None
    truncated: bool

    @property
    def deal_in_count(self) -> int:
        return sum(outcome.focal_deal_in for outcome in self.kyoku_outcomes)

    @property
    def shaped_return(self) -> float:
        return self.raw_return - self.deal_in_penalty - self.placing_penalty


def _validate(
    focal_player: int,
    deal_in_penalty: float,
    placing_penalties: Sequence[float],
    max_steps: int | None,
) -> None:
    if focal_player not in range(4):
        raise ValueError("focal_player must be in 0..3.")
    if len(placing_penalties) != 4:
        raise ValueError("placing_penalties must have four entries, ordered first to fourth.")
    if any(
        not math.isfinite(value) or value < 0 for value in (deal_in_penalty, *placing_penalties)
    ):
        raise ValueError("Penalties must be finite nonnegative magnitudes.")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive.")


def _completed_kyoku(before: Any, after: Any) -> bool:
    # MahJax 0.1.6 no-red auto mode retains terminated_round on a true
    # game end, or replaces RoundState with a fresh one immediately. The
    # round index increases on rotation; honba increases on a dealer repeat.
    old = before.round_state
    new = after.round_state
    return (
        bool(new.terminated_round)
        or int(old.round) != int(new.round)
        or int(old.honba) != int(new.honba)
    )


def _play(
    environment: MahJaxEnvironment,
    policies: Sequence[Policy],
    *,
    seed: int,
    focal_player: int,
    deal_in_penalty: float,
    placing_penalties: Sequence[float],
    max_steps: int | None,
    ron_action: int,
) -> HanchanResult:
    transition = environment.reset(seed)
    score_start = int(environment.state.round_state.score[focal_player])
    kyoku_start = score_start
    raw_return = 0.0
    accrued_deal_in_penalty = 0.0
    outcomes = []
    steps = 0
    final_place = None
    placing_penalty = 0.0
    while not transition.done and (max_steps is None or steps < max_steps):
        before = environment.state
        policy = policies[transition.current_player]
        action = policy.select_action(transition.observation, transition.legal_actions)
        transition = environment.step(action)
        steps += 1
        raw_return += transition.rewards[focal_player]
        if _completed_kyoku(before, environment.state):
            source = None
            chankan = False
            if action == ron_action:
                # No-red settles the first accepted Ron immediately. Inspect
                # the pre-step state; auto-advance may erase these fields.
                source = int(before.round_state.last_player)
                chankan = bool(before.round_state.kan_declared)
            focal_deal_in = source == focal_player and not chankan
            kyoku_end = int(environment.state.round_state.score[focal_player])
            outcomes.append(
                KyokuOutcome(
                    round_index=int(before.round_state.round),
                    honba=int(before.round_state.honba),
                    focal_score_delta=kyoku_end - kyoku_start,
                    ron_source=source,
                    chankan=chankan,
                    focal_deal_in=focal_deal_in,
                )
            )
            kyoku_start = kyoku_end
            if focal_deal_in:
                accrued_deal_in_penalty += deal_in_penalty

    truncated = transition.truncated or not transition.terminated
    if transition.terminated and not truncated:
        ranking = environment.final_ranking()
        if not bool(environment.state.round_state.terminated_round) or ranking is None:
            raise RuntimeError("MahJax terminated without completing a hand and ranking the match.")
        final_place = ranking.index(focal_player) + 1
        placing_penalty = placing_penalties[final_place - 1]

    return HanchanResult(
        focal_player=focal_player,
        steps=steps,
        kyoku_outcomes=tuple(outcomes),
        raw_return=raw_return,
        score_delta=int(environment.state.round_state.score[focal_player]) - score_start,
        deal_in_penalty=accrued_deal_in_penalty,
        placing_penalty=placing_penalty,
        final_place=final_place,
        truncated=truncated,
    )


def _make_environment() -> tuple[MahJaxEnvironment, int]:
    from mahjax.no_red_mahjong.action import Action

    return MahJaxEnvironment(round_mode="half"), int(Action.RON)


def evaluate_hanchan(
    policies: Sequence[Policy],
    *,
    seed: int = 0,
    focal_player: int = 0,
    deal_in_penalty: float = 0.0,
    placing_penalties: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    max_steps: int | None = None,
) -> HanchanResult:
    """Evaluate four seat-indexed policies in one East–South match."""
    if len(policies) != 4:
        raise ValueError("Exactly four seat-indexed policies are required.")
    _validate(focal_player, deal_in_penalty, placing_penalties, max_steps)
    environment, ron_action = _make_environment()
    return _play(
        environment,
        policies,
        seed=seed,
        focal_player=focal_player,
        deal_in_penalty=deal_in_penalty,
        placing_penalties=placing_penalties,
        max_steps=max_steps,
        ron_action=ron_action,
    )


def run_hanchan_evaluation(
    matches: int,
    *,
    seed: int = 0,
    focal_player: int = 0,
    deal_in_penalty: float = 0.0,
    placing_penalties: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    max_steps: int | None = None,
) -> tuple[HanchanResult, ...]:
    """Run a seeded random-policy evaluation, reusing the compiled environment."""
    if matches < 1:
        raise ValueError("matches must be positive.")
    _validate(focal_player, deal_in_penalty, placing_penalties, max_steps)
    environment, ron_action = _make_environment()
    return tuple(
        _play(
            environment,
            [RandomAgent(seed=(seed + match) * 4 + seat) for seat in range(4)],
            seed=seed + match,
            focal_player=focal_player,
            deal_in_penalty=deal_in_penalty,
            placing_penalties=placing_penalties,
            max_steps=max_steps,
            ron_action=ron_action,
        )
        for match in range(matches)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--focal-player", type=int, default=0)
    parser.add_argument("--deal-in-penalty", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int)
    placing = parser.add_mutually_exclusive_group()
    placing.add_argument("--fourth-place-penalty", type=float, default=0.0)
    placing.add_argument("--placing-penalties", type=float, nargs=4, metavar="PENALTY")
    args = parser.parse_args(argv)
    try:
        results = run_hanchan_evaluation(
            args.matches,
            seed=args.seed,
            focal_player=args.focal_player,
            deal_in_penalty=args.deal_in_penalty,
            placing_penalties=args.placing_penalties or (0, 0, 0, args.fourth_place_penalty),
            max_steps=args.max_steps,
        )
    except ValueError as error:
        parser.error(str(error))

    finished = [result for result in results if result.final_place is not None]
    fourths = sum(result.final_place == 4 for result in finished)
    kyoku = sum(len(result.kyoku_outcomes) for result in results)
    deal_ins = sum(result.deal_in_count for result in results)
    print(f"Completed hanchan: {len(finished)}/{len(results)}")
    print(f"Fourth places: {fourths}/{len(finished)}" if finished else "Fourth places: N/A")
    print(f"Discard deal-ins: {deal_ins}/{kyoku}" if kyoku else "Discard deal-ins: N/A")
    if finished:
        mean_raw = sum(result.raw_return for result in finished) / len(finished)
        mean_delta = sum(result.score_delta for result in finished) / len(finished)
        print(f"Mean raw return (completed): {mean_raw:.3f}")
        print(f"Mean score delta (completed): {mean_delta:.3f}")
        print(
            f"Mean shaped return (completed): "
            f"{sum(r.shaped_return for r in finished) / len(finished):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
