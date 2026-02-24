# Claude Session Notes

Persistent notes across Claude Code sessions for the IPW project. Updated after each session.

---

## Session: 2026-02-14 — Parallel Workload Characterization

### What Was Done

Rewrote `run_workload_characterization.py` for parallel execution across 2 vLLM instances (8x A100-80GB, TP=4 each) with ThreadPoolExecutor.

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| Script rewrite | Parallel execution with `--workers`, `--vllm-urls`, streaming JSONL | Done | Full backward compatibility with `--vllm-url` |
| 2nd vLLM instance | Port 8001 on GPUs 4-7 | Running | PID in logs/vllm_serve_8001.log |
| Parallel run (chat, rag, agentic) | 120/120 queries across 2 instances | Done | ~14 min total |
| Reasoning rerun | 40 queries with fixed max_tokens=16384 | Running | Original had max_tokens=32768 = max_model_len, causing 400 errors |

### Architecture Changes

**New CLI args:** `--workers` (default 16), `--vllm-urls` (comma-separated, default localhost:8000), `--limit` (default changed 100→40)

**Execution model:** Load datasets sequentially (HF not thread-safe) → Build WorkItems → ThreadPoolExecutor(max_workers) → Round-robin URLs → Streaming JSONL per query → Convert traces→profiles after all complete

**Key classes:** `ProgressTracker` (thread-safe JSONL writes + progress logging), `WorkItem` (prepared query data for thread pool)

### Bug Fix: Reasoning max_tokens

`WORKLOAD_PARAMS["reasoning"]["max_tokens"]` was 32768, matching `--max-model-len 32768` on vLLM. This caused all reasoning queries to fail with 400 Bad Request (`max_tokens + input_tokens > max_model_len`). Fixed to 16384.

### Data Artifacts

| Path | Contents |
|------|----------|
| `data/active_characterization/workload_traces/` | Per-workload streaming JSONL traces |
| `data/active_characterization/workload_profiles/` | Active WorkloadProfile JSONs (4 workloads) |
| `data/active_characterization/characterization_summary.json` | Run config + timing + results |
| `logs/` | All runtime logs (vLLM, characterization runs) |

### Log Files

| File | Contents |
|------|----------|
| `logs/vllm_serve.log` | vLLM instance 1 (port 8000, GPUs 0-3) |
| `logs/vllm_serve_8001.log` | vLLM instance 2 (port 8001, GPUs 4-7) |
| `logs/characterization_parallel_run1.log` | First parallel run (reasoning all failed) |
| `logs/characterization_reasoning_rerun.log` | Reasoning rerun with fixed max_tokens |
| `logs/characterization_sequential.log` | Old sequential run (killed) |

### Files Modified

- `intelligence-per-watt/src/evals/scripts/run_workload_characterization.py` — Full rewrite for parallel execution
- `.gitignore` — Added `logs/` directory

### Known Issues / Follow-ups

- **Reasoning profile needs merging**: The reasoning rerun output overwrites the (failed) reasoning traces from run 1. Chat/rag/agentic profiles are from run 1.
- **Phase 3 Step 6 pending**: Run full pipeline with active profiles and compare with synthetic results.

---

## Session: 2026-02-13 — End-to-End Pipeline Execution

### What Was Done

Executed the full end-to-end pipeline plan (`plans/witty-beaming-flamingo.md`) on 8x A100-80GB:

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| Step 1 (Pipeline #1a) | GPU profiling of Qwen/Qwen3-8B | Done | 1,562 measurements (token_ops + attention), plus 56 optional (agentic + sampling) |
| Step 2 (Pipeline #1b) | Train estimators & generate LUTs | Done | RidgeRegression selected; R2_train=0.33, R2_val=0.08 |
| Step 3 (Stage 2) | Characterize 5 workload types | Done | chat, reasoning, agentic, rag, coding from real HF datasets |
| Step 4 (Pipeline #2+#3) | Simulate & search per workload | Done | All 5 workloads completed; reasoning produced 0 requests (long outputs exceed 10s sim window) |

### Data Artifacts

| Path | Contents |
|------|----------|
| `data/profiles/Qwen_Qwen3-8B/nvidia_a100_80gb_sxm/fp16/` | token_ops.csv, attention.csv, agentic.csv, sampling.csv |
| `data/luts/` | gpu_token_ops.npz, attention_prefill.npz, attention_decode.npz |
| `data/workload_profiles/` | 5 workload profile JSONs (chat, reasoning, rag, agentic, coding) |
| `data/pipeline_output/{chat,reasoning,agentic,rag,coding}/luts/` | Per-workload pipeline outputs |

### Key Bug Fixes Applied During Execution

1. **AgentData loader** (`datasets/agentdata.py`): Rewrote for `neulab/agent-data-collection` standardized `std` split format. Uses 5 configs: agenttuning_alfworld, agenttuning_db, agenttuning_webshop, orca_agentinstruct, codeactinstruct.

2. **OpenThoughts loader** (`datasets/openthoughts.py`): Fixed to extract query from `conversations` field (not `problem`). Added support for `<|begin_of_thought|>` tags.

3. **CLI workload-type mapping** (`cli.py`, `orchestrator.py`): Registry uses dataset names (wildchat, openthoughts, hotpotqa, agentdata, swebench), but CLI/orchestrator use workload types (chat, reasoning, rag, agentic, coding). Added mapping dicts in both.

4. **LUT generator combined CSV support** (`lut_generator.py`, `sklearn_base.py`): Profiler outputs combined CSVs (token_ops.csv, attention.csv), but LUT generator expected individual CSVs (linear.csv, attention_prefill.csv). Added `operator_name_to_category` mapping and `load_csv_measurements_auto_category()`.

5. **Simulator LUT loading** (`simulator.py`): Fixed to load separate prefill/decode/token_ops LUTs from LUTBundle attributes instead of passing LUTBundle as a Path. Added `_extract_time()` for [time, energy] arrays. Added 100us minimum time floor to prevent event storms with synthetic data.

6. **ML Oracle LUT naming** (`ml_oracle.py`): Support both naming conventions (gpu_attention_prefill.npz and attention_prefill.npz).

### Hardware Key vs Filesystem Slug

Important distinction:
- Registry key: `a100_80gb` (used in CLI commands and `HardwareSpec.from_registry()`)
- Filesystem slug: `nvidia_a100_80gb_sxm` (from `HardwareSpec.name`, used in profile output paths)

### Known Issues / Follow-ups

- **Low estimator R2** (R2_val=0.08): The Ridge estimator trained on combined CSVs has low predictive power. Consider adding more profiling iterations, sweeping more batch sizes, or using per-operator estimators.
- **Reasoning workload 0 requests**: Very long output tokens (~6873 + ~6293 thinking tokens) exceed the 10s simulation window. Increase `duration_s` or reduce output token distribution for reasoning.
- **Test event storms (FIXED)**: 2 tests (test_orchestrator::test_run_all, test_pipeline_e2e::test_full_pipeline_via_orchestrator) were timing out. Root cause: (a) LUT timing fallback returned 0ns when lookups failed; (b) QPS binary search with no SLA constraints went to QPS=1000 creating millions of events. Fixes: 100us minimum time floor on LUT fallback returns, 500k max_events safety limit in simulator event loop, SLA constraints in tests. **All 493 tests now pass.**

### Files Modified (from `main` baseline)

**Dataset Generator:**
- `src/dataset_generator/cli.py` — workload-type mapping
- `src/dataset_generator/datasets/agentdata.py` — std format rewrite
- `src/dataset_generator/datasets/openthoughts.py` — conversations field extraction
- `src/dataset_generator/datasets/hotpotqa.py` — minor fixes
- `src/dataset_generator/datasets/wildchat.py` — minor fixes
- `src/dataset_generator/datasets/swebench.py` — minor fixes
- `src/dataset_generator/pipeline/__init__.py` — exports
- `src/dataset_generator/pipeline/orchestrator.py` — workload-type mapping
- `src/dataset_generator/characterization/` — new directory (all 5 characterizers + registry)

**Inference Simulator:**
- `src/inference_simulator/engine/simulator.py` — LUT loading, timing extraction, min time floor
- `src/inference_simulator/estimator/__init__.py` — exports
- `src/inference_simulator/estimator/sklearn_base.py` — auto-category CSV loading
- `src/inference_simulator/estimator/per_operator_estimator.py` — new
- `src/inference_simulator/estimator/prediction_cache.py` — new
- `src/inference_simulator/types/__init__.py` — exports
- `src/inference_simulator/types/fitted_distribution.py` — new
- `src/inference_simulator/types/workload_profile.py` — new
- `src/inference_simulator/workload/generator.py` — profile-aware generation

**Inference Search:**
- `src/inference_search/cli.py` — minor updates
- `src/inference_search/ml_oracle.py` — LUT naming conventions

**Tests:**
- `tests/dataset_generator/test_dataset_loaders.py` — AgentData std format
- `tests/dataset_generator/test_characterizers.py` — std format helpers
- `tests/dataset_generator/test_cli_estimate.py` — new
- `tests/dataset_generator/test_orchestrator.py` — new
- `tests/test_pipeline_e2e.py` — expanded
- `tests/inference_simulator/` — multiple new test files
- `tests/inference_search/test_ml_search.py` — new

### Branch

`feat/dataset-generator-pipeline1` — all changes are on this branch, not yet merged to `main`.

---

## Session: 2026-02-13 — Phase 2: Active Workload Characterization & Model Registry

### What Was Done

Implemented Phase 2 plan (`plans/witty-beaming-flamingo.md`) using a 6-teammate agent team. All 4 tasks completed with 721 tests passing (0 regressions).

| Task | Description | Status | Tests |
|------|-------------|--------|-------|
| Task 1 | Model registry (17 models, 7 families) | Done | 37 passed |
| Task 2 | Per-operator estimator wiring + per-category R² | Done | 10 passed |
| Task 3 | 4 new eval benchmarks (WildChat, OpenThoughts, HotpotQA, AgentData) + telemetry | Done | 142 passed |
| Task 4 | Trace-to-profile conversion + run orchestration script | Done | 5 passed |
| Integration | Full cross-domain test suite verification | Done | 721 passed total |

### Files Created

**Model Registry:**
- `src/inference_simulator/types/model_registry.py` — 17 models across 7 families (Qwen3 Dense/MoE, Qwen3-Next, GPT-OSS, GLM-4.7, Kimi-K2.5, Kimi-Linear, Moonlight) with `get_model_spec()` and `list_models()` APIs

**Telemetry Layer:**
- `src/evals/telemetry/__init__.py` — Package exports
- `src/evals/telemetry/trace_collector.py` — TurnTrace, QueryTrace dataclasses + TraceCollector (direct vLLM, multi-turn, react, openhands modes)
- `src/evals/telemetry/trace_to_profile.py` — TraceToProfile.convert() maps QueryTrace[] → WorkloadProfile via FittedDistribution.fit()

**Eval Benchmarks (4 new):**
- `src/evals/benchmarks/wildchat/` — Multi-turn chat replay via TraceCollector, completion rate scoring
- `src/evals/benchmarks/openthoughts/` — Single-prompt reasoning with answer extraction (boxed/bold/therefore patterns), exact match scoring
- `src/evals/benchmarks/hotpotqa/` — RAG with context paragraphs, standard EM + token-level F1 scoring
- `src/evals/benchmarks/agentdata/` — Agentic task completion with heuristic + LLM-as-judge scoring

**Run Orchestration:**
- `src/evals/scripts/run_workload_characterization.py` — CLI to run all 5 workloads, output traces/profiles/results

**Tests (6 new files):**
- `tests/inference_simulator/test_model_registry.py` (37 tests)
- `tests/inference_simulator/test_per_operator_pipeline.py` (10 tests)
- `tests/evals/test_trace_collector.py`, `test_trace_to_profile.py`
- `tests/evals/test_wildchat_benchmark.py`, `test_openthoughts_benchmark.py`, `test_hotpotqa_benchmark.py`, `test_agentdata_benchmark.py`

### Files Modified

- `src/agents/mcp/vllm_server.py` — Added GLM-4.7-Flash to SUPPORTED_MODELS and MODEL_COSTS
- `src/inference_simulator/estimator/model_comparison.py` — Added `compare_estimators_by_category()` for per-category R² breakdown
- `src/inference_simulator/estimator/lut_generator.py` — Added PerOperatorEstimator to candidate list, widened type hints to BaseRuntimeEstimator
- `src/dataset_generator/cli.py` — Added `--per-category` flag to compare-estimators command
- `src/evals/benchmarks/__init__.py` — Updated docstring with new benchmarks
- `src/inference_simulator/types/__init__.py` — Added model registry exports

### Key Design Decisions

- **PerOperatorEstimator extends BaseRuntimeEstimator** (not SklearnEstimatorBase): Required adapter logic in lut_generator.py to handle both estimator base types for LUT generation.
- **TraceCollector modes**: `run_query_direct_vllm()` and `run_query_multi_turn_vllm()` are fully implemented; `run_query_react()` and `run_query_openhands()` are stubs ready for integration.
- **Benchmarks follow existing patterns**: DatasetBenchmark base class with @register_benchmark decorator, auto-discovered via evals/registry.py.

### Known Issues

- **3 pre-existing test failures** in `tests/evals/test_scorers.py`: browsecomp (missing `Optional` import), deepresearch/apex (unimplemented scorers). Not from our changes.
- **Phase 3 pending**: vLLM serving + actual workload characterization runs (requires GPU availability and model download).
