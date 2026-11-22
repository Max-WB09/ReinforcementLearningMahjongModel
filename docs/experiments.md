# Experiments

## Implemented initial experiment

The initial experiment runs four seeded random agents in MahJax `no_red_mahjong` with `round_mode="single"`. An episode is one four-player hand. Each policy selects uniformly among only the legal action IDs supplied with its player-visible observation.

Run it with `python -m mahjong.simulation --episodes 100 --seed 42` after installation. The runner reports episode count, player-0 mean undiscounted return, wall-clock duration, aggregate environment-step throughput, final ranking, and final scores for the final episode. MahJax scores are reported in its hundreds-of-points units. The runner does not estimate unavailable statistics or claim a benchmark.

## Risk-sensitive validation protocol

Use two evaluation units. A **kyoku** is one hand, including a repeated dealer hand; a **hanchan** is a complete East–South match under the simulator's termination rules, which may end early or extend into West. Keep single-hand rollouts for the baseline and the initial training objective. Evaluate frozen policies in a separate hanchan run; this does not require training on entire matches.

| Signal | Unit | Definition |
| --- | --- | --- |
| Raw score change and lower-tail CVaR | Kyoku | Record the focal player's score change before experimental penalties. The final hand’s delta includes any match-end riichi-stick settlement under the simulator rules. Use the same lower-tail fraction for both policies and report it with the result. |
| Discard deal-in | Kyoku | Count at most one event in a completed hand if the focal player's discard was accepted for ron. Do not count another player's discard, tsumo, or a draw. Keep chankan (robbing a kan) separate from discard deal-in. |
| Fourth-place rate | Hanchan | Divide completed matches where the focal player finishes fourth by all completed matches. Rank by score, then by initial seat wind on ties, not player ID or current seat wind. |
| Experimental placement penalty | Hanchan | Apply the configured penalty for the focal player's final place once, after true match termination. Do not apply it after an intermediate hand or if the match is truncated. |

Keep raw measures and shaped returns separate. Penalty magnitudes are experimental configuration, not game scoring rules. Use the same opponent pool, matched evaluation seeds and starting-seat schedule, and the same MahJax version and ruleset for both frozen policies. Report the numbers of completed hands, completed matches, and truncations. Compare fourth-place rates across matches and deal-in rates across hands rather than mixing the two denominators. Use held-out evaluation seeds and the same definition of completed games for each policy.

### Running the hanchan evaluator

Use the separate CPU-compatible random-policy smoke run:

```bash
python -m mahjong.evaluation --matches 3 --seed 42 --focal-player 0 \
  --deal-in-penalty 5 --fourth-place-penalty 20
```

Here 5 and 20 are illustrative magnitudes, not thesis defaults. Both penalties default to zero and use MahJax units (one unit represents 100 game points). Alternatively, pass four magnitudes with `--placing-penalties P1 P2 P3 P4`. For a completed match:

```text
shaped return = raw simulator return - (deal-in count × deal-in penalty)
                - penalty indexed by final place
```

The focal player receives a deal-in charge for each qualifying completed hand, including in an otherwise truncated match. A truncated match never has a final place or placing charge; do not mix its partial returns into completed-match averages. The CLI is a random-policy smoke test only. To evaluate trained policies, pass four seat-indexed policies to `evaluate_hanchan` with the same player-visible observation and legal-action interface as the baseline. These penalties are evaluation outputs; this change does not train a policy or prescribe a PPO objective.

## Reproducibility

Record the command, seed, package versions, JAX backend, device information, and hardware details with each result. Identical seeds are expected to reproduce action selection and MahJax key sequences under the same software and hardware configuration; cross-platform bitwise equality is not asserted.

## Planned work

The random baseline validates lifecycle and measurement plumbing only. Future experiments can add matched expected-return policies, risk-sensitive reward objectives, adverse-outcome definitions, and CVaR-based evaluation or training. Vectorized rollouts should be evaluated with separate CPU/GPU throughput, compilation-time, and memory measurements before being used for thesis conclusions.
