import pytest

from mahjong.environment import IllegalActionError, MahJaxEnvironment


def test_reset_exposes_legal_actions_and_player_observation() -> None:
    environment = MahJaxEnvironment(seed=7)
    transition = environment.reset()

    assert transition.legal_actions
    assert transition.current_player in range(4)
    assert isinstance(transition.observation, dict)
    assert "hand" in transition.observation


def test_legal_action_advances_environment() -> None:
    environment = MahJaxEnvironment(seed=7)
    transition = environment.reset()
    next_transition = environment.step(transition.legal_actions[0])

    assert environment.state.step_count == 1
    assert len(next_transition.rewards) == 4


def test_invalid_action_is_rejected_before_framework_penalty() -> None:
    environment = MahJaxEnvironment(seed=7)
    transition = environment.reset()
    invalid = next(action for action in range(100) if action not in transition.legal_actions)

    with pytest.raises(IllegalActionError):
        environment.step(invalid)


def test_reset_seed_is_reproducible() -> None:
    first = MahJaxEnvironment(seed=9).reset()
    second = MahJaxEnvironment(seed=9).reset()

    assert first.legal_actions == second.legal_actions
    assert first.current_player == second.current_player
