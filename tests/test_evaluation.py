from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from mahjong.environment import StepResult
from mahjong.evaluation import _play, evaluate_hanchan, run_hanchan_evaluation

RON = 8
TSUMO = 9
OTHER = 10
OBSERVATION = {"visible": True}


@dataclass
class ScriptedState:
    round_state: SimpleNamespace
    terminated: bool = False
    truncated: bool = False


def _state(round_index, honba, score, *, payer=-1, chankan=False, ended=False, truncated=False):
    return ScriptedState(
        round_state=SimpleNamespace(
            round=round_index,
            honba=honba,
            score=(500, score, 400, 300),
            last_player=payer,
            kan_declared=chankan,
            terminated_round=ended,
        ),
        terminated=ended,
        truncated=truncated,
    )


class ScriptedEnvironment:
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.index = 0

    @property
    def state(self):
        return self.states[self.index]

    def _transition(self):
        return StepResult(
            OBSERVATION,
            (self.actions[self.index],) if self.index < len(self.actions) else (),
            (0, self.rewards[self.index - 1] if self.index else 0, 0, 0),
            self.state.terminated,
            self.state.truncated,
            2,
        )

    def reset(self, seed):
        self.index = 0
        return self._transition()

    def step(self, action):
        assert action == self.actions[self.index]
        self.index += 1
        return self._transition()

    def final_ranking(self):
        assert self.state.terminated and not self.state.truncated
        return (0, 2, 3, 1)


class FirstLegalPolicy:
    def select_action(self, observation, legal_actions):
        assert observation is OBSERVATION
        return legal_actions[0]


def _evaluate(states, actions, rewards, *, max_steps=None):
    return _play(
        ScriptedEnvironment(states, actions, rewards),
        [FirstLegalPolicy()] * 4,
        seed=42,
        focal_player=1,
        deal_in_penalty=5,
        placing_penalties=(0, 0, 0, 20),
        max_steps=max_steps,
        ron_action=RON,
    )


def test_deal_in_is_counted_per_kyoku_and_placing_once_at_match_end():
    result = _evaluate(
        [
            _state(0, 0, 250, payer=1),
            _state(0, 1, 240),
            _state(0, 1, 240, payer=1, chankan=True),
            _state(1, 0, 220),
            _state(2, 0, 215, payer=3),
            _state(3, 0, 215, payer=1),
            _state(3, 0, 212, ended=True),
        ],
        [RON, OTHER, RON, TSUMO, RON, RON],
        [-10, 0, -20, -5, 0, -3],
    )
    outcomes = result.kyoku_outcomes
    assert [(o.round_index, o.honba) for o in outcomes] == [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
    assert [o.ron_source for o in outcomes] == [1, 1, None, 3, 1]
    assert [o.chankan for o in outcomes] == [False, True, False, False, False]
    assert [o.focal_score_delta for o in outcomes] == [-10, -20, -5, 0, -3]
    assert result.deal_in_count == 2
    assert result.deal_in_penalty == 10
    assert result.final_place == 4
    assert result.placing_penalty == 20
    assert result.raw_return == result.score_delta == -38
    assert result.shaped_return == -68
    assert not result.truncated


@pytest.mark.parametrize("simulator_truncated", [False, True])
def test_truncated_match_never_gets_placing_penalty(simulator_truncated):
    result = _evaluate(
        [_state(0, 0, 250, payer=1), _state(0, 1, 240, truncated=simulator_truncated)],
        [RON],
        [-10],
        max_steps=None if simulator_truncated else 1,
    )
    assert result.truncated
    assert result.final_place is None
    assert result.placing_penalty == 0
    assert result.deal_in_count == 1
    assert result.shaped_return == -15


def test_zero_defaults_and_score_accounting_with_pinned_mahjax():
    (result,) = run_hanchan_evaluation(1, seed=42)
    assert not result.truncated
    assert result.final_place in range(1, 5)
    assert len(result.kyoku_outcomes) >= 2
    assert sum(outcome.focal_score_delta for outcome in result.kyoku_outcomes) == result.score_delta
    assert result.shaped_return == result.raw_return
    assert result.placing_penalty == 0
    assert result.deal_in_penalty == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"focal_player": 4},
        {"deal_in_penalty": -1},
        {"placing_penalties": (0, 0, 0)},
        {"placing_penalties": (0, 0, 0, float("nan"))},
        {"max_steps": 0},
    ],
)
def test_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        evaluate_hanchan([FirstLegalPolicy()] * 4, **kwargs)
