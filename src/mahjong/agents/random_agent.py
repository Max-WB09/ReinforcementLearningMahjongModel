"""A reproducible random policy over an already legal action set."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


class RandomAgent:
    """Choose uniformly from supplied legal actions without inspecting game state."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def select_action(
        self, observation: Mapping[str, Any], legal_actions: Sequence[int]
    ) -> int:
        """Select an action using only player observation and legal action IDs."""
        del observation
        if not legal_actions:
            raise ValueError("Cannot select an action from an empty legal action set.")
        return self._random.choice(list(legal_actions))
