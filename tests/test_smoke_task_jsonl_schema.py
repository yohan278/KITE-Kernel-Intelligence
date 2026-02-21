import json
import os
import subprocess
from pathlib import Path

from kite.utils.serialization import load_jsonl


def test_smoke_jsonl_schema(tmp_path: Path) -> None:
    cfg = tmp_path / "smoke.yaml"
    out_jsonl = tmp_path / "smoke.jsonl"
    trace_dir = tmp_path / "traces"
    kb_root = tmp_path / "KernelBench"
    kb_root.mkdir(parents=True, exist_ok=True)

    cfg.write_text(
        "\n".join(
            [
                f"kernelbench_root: {kb_root}",
                f"output_jsonl: {out_jsonl}",
                f"power_trace_dir: {trace_dir}",
                "task_index: 0",
                "warmup_iters: 1",
                "measure_iters: 2",
                "repeats: 1",
                "sampling_interval_ms: 5.0",
            ]
        )
    )

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "smoke_test_one_task.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run(["python", str(script), "--config", str(cfg)], check=True, cwd=repo, env=env)

    rows = load_jsonl(out_jsonl)
    assert len(rows) == 1
    row = rows[0]
    required = {
        "task_id",
        "compile_ok",
        "correct",
        "runtime_ms",
        "avg_watts",
        "joules",
        "speedup_vs_baseline",
        "logs_path",
    }
    assert required.issubset(row.keys())
    assert row["runtime_ms"] is not None
    assert row["joules"] is not None

    trace_path = Path(row["logs_path"])
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text())
    assert "samples" in payload
