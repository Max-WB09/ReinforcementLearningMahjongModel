"""Small, public interface around the verified MahJax 0.1.6 environment API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class IllegalActionError(ValueError):
    """Raised before an action outside MahJax's legal-action mask is applied."""


@dataclass(frozen=True)
class StepResult:
    """Public transition data, without exposing privileged game state."""

    observation: Mapping[str, Any]
    legal_actions: tuple[int, ...]
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    current_player: int

    @property
    def done(self) -> bool:
        """Whether the episode is terminal or externally truncated."""
        return self.terminated or self.truncated


class MahJaxEnvironment:
    """Stateful adapter for one unbatched MahJax episode.

    The default is MahJax's ``no_red_mahjong`` with ``round_mode='single'``:
    one four-player Riichi Mahjong hand. It is deliberately not an East-only,
    half-game, or complete match. Observations are obtained only through
    ``env.observe``; this adapter never calls MahJax's privileged-observation
    interface.
    """

    def __init__(
        self,
        *,
        environment_id: str = "no_red_mahjong",
        round_mode: str = "single",
        seed: int | None = None,
    ) -> None:
        try:
            import jax
            import mahjax
        except ImportError as error:
            raise ImportError(
                "MahJax and JAX are required. Install the project dependencies first."
            ) from error

        if round_mode not in {"single", "east", "half"}:
            raise ValueError("round_mode must be one of: 'single', 'east', or 'half'.")
        if environment_id not in {"no_red_mahjong", "red_mahjong"}:
            raise ValueError("environment_id must be 'no_red_mahjong' or 'red_mahjong'.")

        self._jax = jax
        self._env = mahjax.make(
            environment_id,
            round_mode=round_mode,
            observe_type="dict",
            order_points=[0, 0, 0, 0],
        )
        # MahJax's action dispatch is designed for JAX tracing. JIT also avoids
        # eagerly evaluating inactive action branches on recent JAX releases.
        self._init = jax.jit(self._env.init)
        self._step = jax.jit(self._env.step)
        self._observe = jax.jit(self._env.observe)
        self._key = jax.random.PRNGKey(0 if seed is None else seed)
        self._state: Any | None = None
        self.environment_id = environment_id
        self.round_mode = round_mode

    @property
    def state(self) -> Any:
        """Current raw MahJax state for diagnostics; do not give it to policies."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing the state.")
        return self._state

    def reset(self, seed: int | None = None) -> StepResult:
        """Reset one episode, optionally replacing the deterministic PRNG seed."""
        if seed is not None:
            self._key = self._jax.random.PRNGKey(seed)
        self._key, init_key = self._jax.random.split(self._key)
        self._state = self._init(init_key)
        return self._result()

    def legal_actions(self) -> tuple[int, ...]:
        """Return action IDs permitted for the current player by MahJax."""
        return tuple(int(value) for value in self._jax.numpy.where(self.state.legal_action_mask)[0])

    def step(self, action: int) -> StepResult:
        """Apply a legal action and return the next player-visible transition."""
        if self.is_done:
            raise RuntimeError("Cannot step a completed episode; call reset() first.")
        if action not in self.legal_actions():
            raise IllegalActionError(f"Action {action} is not legal in the current state.")
        self._key, step_key = self._jax.random.split(self._key)
        self._state = self._step(self.state, action, step_key)
        return self._result()

    @property
    def is_done(self) -> bool:
        """Whether the current episode has ended."""
        return bool(self.state.terminated) or bool(self.state.truncated)

    def final_scores(self) -> tuple[int, ...] | None:
        """Return MahJax final scores (hundreds of points), once the episode ends."""
        if not self.is_done:
            return None
        return tuple(int(value) for value in self.state.round_state.score)

    def final_ranking(self) -> tuple[int, ...] | None:
        """Return player IDs ordered by descending final MahJax score."""
        scores = self.final_scores()
        if scores is None:
            return None
        return tuple(sorted(range(len(scores)), key=lambda player: (-scores[player], player)))

    def _result(self) -> StepResult:
        observation = self._observe(self.state)
        return StepResult(
            observation=observation,
            legal_actions=self.legal_actions(),
            rewards=tuple(float(value) for value in self.state.rewards),
            terminated=bool(self.state.terminated),
            truncated=bool(self.state.truncated),
            current_player=int(self.state.current_player),
        )
