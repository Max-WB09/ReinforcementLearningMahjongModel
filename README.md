# Risk-Sensitive Reinforcement Learning under Imperfect Information

This undergraduate thesis repository studies the trade-off between expected return and adverse outcomes in Riichi Mahjong. It currently provides a reproducible, CPU-compatible MahJax simulation baseline; it does **not** yet train PPO, CVaR, or neural-network policies.

MahJax was selected because it is a JAX-native, four-player Riichi Mahjong environment with explicit legal-action masks and a documented vectorized API. The initial baseline uses its simplified `no_red_mahjong` ruleset in `single` mode: one four-player hand, not a complete match.

## Quick start

Python 3.10--3.13 is supported. Create an environment, then install the project and its development tools:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m mahjong.simulation --episodes 10 --seed 42
```

Run the test suite with:

```bash
pytest
```

Detailed material: [MahJax motivation](docs/mahjax_motivation.md), [architecture](docs/architecture.md), [environment setup](docs/environment_setup.md), and [experiments](docs/experiments.md).

## Planned milestones

1. Establish random-policy simulation measurements.
2. Add expected-return policy training.
3. Add risk-sensitive objectives and adverse-outcome evaluation.
4. Evaluate future CVaR-based methods under matched experimental conditions.
