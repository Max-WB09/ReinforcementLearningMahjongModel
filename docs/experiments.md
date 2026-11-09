# Experiments

## Implemented initial experiment

The initial experiment runs four seeded random agents in MahJax `no_red_mahjong` with `round_mode="single"`. An episode is one four-player hand. Each policy selects uniformly among only the legal action IDs supplied with its player-visible observation.

Run it with `python -m mahjong.simulation --episodes 100 --seed 42` after installation. The runner reports episode count, player-0 mean undiscounted return, wall-clock duration, aggregate environment-step throughput, final ranking, and final scores for the final episode. MahJax scores are reported in its hundreds-of-points units. The runner does not estimate unavailable statistics or claim a benchmark.

## Reproducibility

Record the command, seed, package versions, JAX backend, device information, and hardware details with each result. Identical seeds are expected to reproduce action selection and MahJax key sequences under the same software and hardware configuration; cross-platform bitwise equality is not asserted.

## Planned work

The random baseline validates lifecycle and measurement plumbing only. Future experiments can add matched expected-return policies, risk-sensitive reward objectives, adverse-outcome definitions, and CVaR-based evaluation or training. Vectorized rollouts should be evaluated with separate CPU/GPU throughput, compilation-time, and memory measurements before being used for thesis conclusions.
