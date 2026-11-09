from mahjong.simulation import run_simulation
from mahjong.utils.device import get_device_info


def test_simulation_completes_single_episode() -> None:
    summary = run_simulation(1, seed=3)
    result = summary.episodes[0]

    assert result.steps > 0
    assert len(result.returns) == 4
    assert result.final_scores is not None
    assert result.final_ranking is not None
    assert summary.throughput_steps_per_second > 0


def test_cpu_compatible_device_inspection() -> None:
    info = get_device_info()

    assert info.jax_version
    assert info.devices
    assert info.default_backend in {"cpu", "gpu", "tpu"}
