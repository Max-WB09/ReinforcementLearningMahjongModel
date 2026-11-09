# Why MahJax for This Thesis

## Research context

Riichi Mahjong is a four-player, stochastic, imperfect-information game. A player must act from public information and their own concealed hand while accounting for opponents' unknown hands, future draws, score changes, and potentially costly decisions. These features make it a useful setting for comparing policies that maximize expected return with policies that explicitly limit undesirable outcomes.

## Why MahJax?

MahJax is a JAX implementation of Riichi Mahjong intended for reinforcement-learning research. Its documented interface provides explicit state initialization, transitions, player observations, legal-action masks, and JAX transformations such as `jit` and `vmap`. The project documentation describes both standard red-five and simplified no-red four-player rulesets, plus `single`, East-only, and half-game round modes. The environment's PRNG-key interface also makes seeded simulation reproducible when package and hardware versions are recorded.

This repository uses `no_red_mahjong`, `round_mode="single"`, and dictionary observations for the initial baseline. It therefore simulates one four-player hand, not an East-only game, half-game, or full match. The adapter supplies an agent only with `env.observe(state)` and the current legal-action mask; it deliberately does not call MahJax's privileged-observation API.

MahJax documents vectorized execution through `jax.vmap` and compilation through `jax.jit`. This foundation is promising for future batched rollouts on CPU or GPU, but this repository does not claim a local speedup: no benchmark has been performed here. The current sequential runner JIT-compiles single-environment initialization, stepping, and observation; a vectorized runner should be introduced only after measuring compilation time, throughput, and memory use. On an 8 GB GPU, small batches with measured memory consumption are the appropriate starting point.

## Comparison with RiichiEnv

| Aspect | MahJax | RiichiEnv |
| --- | --- | --- |
| Primary ecosystem | Python/JAX | Python interface with Rust core |
| Public style | Pgx-like functional `init`, `step`, `observe` API | Gym-style `reset`, `step` API |
| Acceleration model | JAX compilation and vectorization; GPU possible with a compatible JAX build | Native Rust transitions; its public materials emphasize high performance |
| Integration cost here | Direct fit with planned JAX-based RL work | Would require a different API and likely a separate interoperability layer |

These are verified design differences from the projects' public documentation, not a benchmark. Neither project is asserted to be faster in this repository. RiichiEnv may be a useful future comparison baseline, especially where its ruleset flexibility or Rust implementation matters.

## Research relevance

The same seeded environment configuration and action interface can support an expected-return policy, reward transformations or penalties for adverse events, and later distributional or CVaR-oriented optimization. Episode rewards, terminal scores, and player rankings are collected only when exposed by MahJax. This makes controlled policy comparisons possible without reimplementing Mahjong rules.

## Limitations and risks

MahJax 0.1.6 is under active development and labels its API as provisional. Its observation features are also documented as not finalized. The selected no-red environment intentionally omits red fives, special abortive draws, pao, and double ron; it is a baseline environment rather than a complete representation of all Riichi rules. Full-match modes can be longer and more resource-intensive. GPU memory, CUDA/JAX compatibility, API changes, available terminal statistics, and the difficulty of training strong Mahjong policies all remain experimental risks.

## Conclusion

MahJax is an evidence-based choice for this thesis because it offers a documented JAX interface, legal-action masking, seeded state transitions, and a route to later vectorized rollouts. The decision is conditional on continuing API validation and explicit performance measurement, not on an unverified local performance claim.

## Sources

- [MahJax source repository and API examples](https://github.com/nissymori/mahjax)
- [MahJax no-red rules documentation](https://nissymori.github.io/mahjax/no_red_mahjong/)
- [MahJax paper (arXiv:2605.20577)](https://arxiv.org/abs/2605.20577)
- [RiichiEnv source repository](https://github.com/smly/RiichiEnv)
