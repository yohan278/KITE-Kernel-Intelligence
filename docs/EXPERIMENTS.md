# KITE Experiments

## Phases

1. Baselines
- KernelBench default outputs.
- IPW telemetry validation on fixed kernels.

2. SFT
- Build prompt/kernel pairs from valid trajectories.

3. Kernel GRPO
- Group rollouts.
- Correctness-heavy early epochs.
- Speedup-biased later epochs.

4. Energy-Aware Kernel GRPO
- Add energy/token and power penalties.
- Enforce hard penalty for incorrect kernels.

5. Runtime PPO
- Optimize power cap, DVFS profile, microbatch, concurrency.

6. Hierarchical RL
- Alternate kernel and runtime updates.

7. Final Evaluation
- Run B0/B1/E1/E2/E3/E4 and A1/A2/A3.
- Generate Pareto curves and SLA analysis.

## Acceptance Targets

- Correctness >= 95% of performance-only baseline.
- Decode energy/token improvement >= 10% at matched correctness.
- APJ/APW improvement at matched p95 latency SLA.
- SLA violation rate < 5%.
