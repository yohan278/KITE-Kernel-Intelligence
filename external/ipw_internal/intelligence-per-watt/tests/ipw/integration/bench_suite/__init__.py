"""Integration test suite for ipw bench CLI.

This module provides a parametrized integration test suite that runs all combinations
of LM-agent-benchmark-resource tuples on real hardware.

Test Matrix (24 total combinations):
- LMs: Qwen3 8B, GPT OSS 20B
- Agents: OpenHands, ReAct, Orchestrator
- Benchmarks: HLE, GAIA
- Resources: 1 GPU/8 CPUs, 4 GPUs/32 CPUs

Total: 2 x 3 x 2 x 2 = 24 test combinations
"""
