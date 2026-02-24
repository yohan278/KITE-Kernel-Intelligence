#!/usr/bin/env python3
"""Training Pipeline: Sweep -> Best Checkpoint -> Eval on all benchmarks.

Simple GPU orchestration: user specifies which GPUs to use via --gpus flag.
Sweep agents run on all GPUs in parallel, then benchmarks are distributed
across the same GPUs (one benchmark per GPU at a time).

Usage:
    python training/pipeline.py \
        --dataset ./data/filtered_traces.parquet \
        --model Qwen/Qwen3-8B \
        --paradigm LORA \
        --max-runs 12 \
        --gpus 0,1,2,3,4,5,6,7 \
        --project my-sweep-project
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

# =============================================================================
# Constants
# =============================================================================

BENCHMARKS = ["hle", "gaia", "simpleqa", "deepresearch"]

PARADIGM_SCRIPTS = {
    "LORA": "training/scripts/sweep_lora_hparams.py",
}

ORCHESTRATOR_ROOT = Path(__file__).parent.parent  # orchestrator/

# =============================================================================
# Process Tracking & Graceful Shutdown
# =============================================================================

_tracked_processes: List[Tuple[subprocess.Popen, Optional[object], str]] = []
_shutdown_in_progress = False
_sweep_output_dir: Optional[Path] = None


def _cleanup_all_processes():
    """Terminate all tracked subprocesses."""
    global _shutdown_in_progress
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    if not _tracked_processes:
        return

    log("[Shutdown] Terminating all running subprocesses...")
    for proc, lh, label in _tracked_processes:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    deadline = time.time() + 5
    for proc, lh, label in _tracked_processes:
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass

    for _, lh, _ in _tracked_processes:
        if lh and not lh.closed:
            try:
                lh.close()
            except Exception:
                pass

    _tracked_processes.clear()
    log("[Shutdown] Done.")


def _cleanup_artifacts_on_interrupt():
    """Remove sweep artifacts if no models were successfully trained yet.

    Keeps artifacts if any of the following exist:
    - best_model_info.json or best/ directory (completed run saved a best model)
    - Any checkpoint-* directory under any run directory (mid-training progress)
    """
    if _sweep_output_dir is None:
        return

    best_dir = _sweep_output_dir / "best"
    best_info = _sweep_output_dir / "best_model_info.json"

    has_trained_models = best_info.exists() or (
        best_dir.exists() and any(best_dir.iterdir())
    )

    has_checkpoints = any(_sweep_output_dir.rglob("checkpoint-*"))

    if not has_trained_models and not has_checkpoints:
        import shutil
        log("[Shutdown] No models trained and no checkpoints found — removing incomplete sweep artifacts...")
        try:
            shutil.rmtree(_sweep_output_dir)
            log(f"[Shutdown] Removed {_sweep_output_dir}")
        except Exception as e:
            log(f"[Shutdown] Warning: could not remove artifacts: {e}")
    elif has_checkpoints and not has_trained_models:
        log(f"[Shutdown] Checkpoints found — keeping sweep artifacts at {_sweep_output_dir}")


def _signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received {sig_name}. Shutting down...")
    _cleanup_all_processes()
    _cleanup_artifacts_on_interrupt()
    sys.exit(130 if signum == signal.SIGINT else 143)


# =============================================================================
# Log Archival
# =============================================================================


def _archive_logs(log_dir: Path):
    """Move existing log files into a timestamped subdirectory.

    Called before launching new agents so that logs from previous (possibly
    failed) runs are preserved for debugging instead of being appended to or
    silently lost.  The archive is placed under ``log_dir/archive/<timestamp>/``.
    """
    if not log_dir.exists():
        return
    log_files = [f for f in log_dir.iterdir() if f.is_file() and f.suffix == ".log"]
    if not log_files:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = log_dir / "archive" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)

    for lf in log_files:
        lf.rename(archive_dir / lf.name)

    log(f"[Logs] Archived {len(log_files)} old log file(s) → {archive_dir}")


# =============================================================================
# Agent State Tracking
# =============================================================================


@dataclass
class _AgentState:
    """Tracks the state of a single sweep agent for dashboard display."""

    gpu_id: int
    label: str
    status: str = "starting"  # starting | loading | training | idle | done | failed
    config: dict = field(default_factory=dict)
    start_time: float = 0.0
    finish_time: Optional[float] = None
    error: Optional[str] = None
    run_count: int = 0


def _parse_sweep_line(line: str, state: _AgentState):
    """Parse a log line from a sweep agent to update its dashboard state."""
    s = line.strip()
    if not s:
        return
    if "LoRA Training Run" in s:
        state.status = "loading"
        state.config = {}
    elif "LoRA Rank:" in s and "values" not in s:
        try:
            state.config["rank"] = int(s.split(":")[-1].strip())
        except (ValueError, IndexError):
            pass
    elif "Learning Rate:" in s:
        try:
            state.config["lr"] = s.split(":")[-1].strip()
        except (ValueError, IndexError):
            pass
    elif s.startswith("Epochs:"):
        try:
            state.config["epochs"] = int(s.split(":")[-1].strip())
        except (ValueError, IndexError):
            pass
    elif "Starting Training" in s:
        state.status = "training"
    elif "Training Complete" in s:
        state.run_count += 1
        state.status = "idle"
    elif "Error during training" in s:
        state.status = "failed"
        state.error = s[:200]
    elif "OutOfMemoryError" in s or "CUDA out of memory" in s:
        state.status = "failed"
        state.error = "CUDA out of memory"


# =============================================================================
# Helpers
# =============================================================================


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _tail_log_file(path: Path, n_lines: int = 30) -> str:
    if not path.exists():
        return "(log file not found)"
    try:
        with open(path) as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        return "".join(tail)
    except Exception as e:
        return f"(could not read log: {e})"


def _start_log_streamer(proc: subprocess.Popen, log_path: Path, prefix: str,
                         agent_state: Optional[_AgentState] = None):
    """Daemon thread that writes subprocess stdout to log file.

    When the rich dashboard is active, output goes to log file only (no terminal
    noise). If agent_state is provided, parses lines to update the dashboard.
    """
    def _stream():
        try:
            with open(log_path, "a") as lf:
                for raw_line in iter(proc.stdout.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace")
                    lf.write(line)
                    lf.flush()
                    if agent_state is not None:
                        _parse_sweep_line(line, agent_state)
                    if not _RICH:
                        stripped = line.rstrip("\n")
                        if stripped:
                            print(f"  [{prefix}] {stripped}", flush=True)
        except Exception:
            pass

    t = threading.Thread(target=_stream, daemon=True)
    t.start()


def _launch_subprocess(
    cmd: List[str], env: dict, log_path: Path, label: str,
    agent_state: Optional[_AgentState] = None,
) -> subprocess.Popen:
    """Launch a tracked subprocess with output streamed to log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    _start_log_streamer(proc, log_path, label, agent_state=agent_state)
    _tracked_processes.append((proc, None, label))
    return proc


def _untrack_process(proc: subprocess.Popen):
    for i, (p, lh, _) in enumerate(_tracked_processes):
        if p is proc:
            if lh and not lh.closed:
                lh.close()
            _tracked_processes.pop(i)
            return


def _report_failure(label: str, returncode: int, log_path: Path,
                    errors_list: Optional[List[str]] = None):
    """Report a subprocess failure.

    If errors_list is provided, appends a one-line summary (for the dashboard
    errors panel). Otherwise, prints detailed log tail to terminal (fallback).
    """
    tail = _tail_log_file(log_path, n_lines=15)

    # Extract the most relevant error line
    error_detail = ""
    for tail_line in reversed(tail.splitlines()):
        stripped = tail_line.strip()
        if stripped and any(kw in stripped.lower()
                           for kw in ["error", "exception", "oom", "killed", "cuda"]):
            error_detail = stripped[:150]
            break

    summary = f"{label} (exit code {returncode})"
    if error_detail:
        summary += f" — {error_detail}"

    if errors_list is not None:
        errors_list.append(summary)
    else:
        print(flush=True)
        log(f"ERROR: {label} failed with exit code {returncode}")
        log(f"Last output from {log_path}:")
        print("-" * 60)
        print(tail)
        print("-" * 60)
        print(flush=True)


# =============================================================================
# Dashboard (Rich)
# =============================================================================


def _build_progress_bar(done: int, total: int, width: int = 40) -> "Text":
    """Build a colored text-based progress bar."""
    pct = done / total if total > 0 else 0
    filled = int(width * pct)
    text = Text("  ")
    text.append("━" * filled, style="green bold")
    text.append("─" * (width - filled), style="dim")
    text.append(f"  {done}/{total}  {pct:.0%}", style="bold")
    return text


def _build_sweep_display(
    agent_states: Dict[int, _AgentState],
    sweep_status: Dict,
    max_runs: int,
    elapsed_sec: float,
    errors: List[str],
    log_dir: Optional[Path] = None,
    sweep_url: Optional[str] = None,
) -> "Panel":
    """Build a rich Panel showing sweep progress, agent table, and errors."""
    done = sweep_status.get("finished", 0)
    running = sweep_status.get("running", 0)
    failed = sweep_status.get("failed", 0)

    # Header stats
    header = Text("  ")
    header.append("Completed ", style="dim")
    header.append(str(done), style="bold green" if done > 0 else "bold")
    header.append(f"/{max_runs}  ", style="dim")
    if running:
        header.append("Running ", style="dim")
        header.append(str(running), style="bold yellow")
        header.append("  ", style="dim")
    if failed:
        header.append("Failed ", style="dim")
        header.append(str(failed), style="bold red")
        header.append("  ", style="dim")
    header.append("Elapsed ", style="dim")
    header.append(fmt_duration(elapsed_sec), style="bold")

    progress = _build_progress_bar(done, max_runs)

    # Agent table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("GPU", style="dim", width=5, justify="center")
    table.add_column("Status", width=14)
    table.add_column("Config", width=30)
    table.add_column("Runs", width=5, justify="center")
    table.add_column("Elapsed", width=10, justify="right")

    status_styles = {
        "starting": ("◌ Starting", "dim"),
        "loading":  ("◌ Loading", "blue"),
        "training": ("● Training", "yellow"),
        "idle":     ("◌ Waiting", "cyan"),
        "done":     ("✓ Done", "green"),
        "failed":   ("✗ Failed", "red"),
    }

    for gpu_id in sorted(agent_states):
        st = agent_states[gpu_id]
        s_text, s_style = status_styles.get(st.status, ("?", "dim"))
        status = Text(s_text, style=s_style)

        cfg = st.config
        if cfg:
            parts = []
            if "rank" in cfg:
                parts.append(f"r={cfg['rank']}")
            if "lr" in cfg:
                parts.append(f"lr={cfg['lr']}")
            if "epochs" in cfg:
                parts.append(f"ep={cfg['epochs']}")
            config_str = "  ".join(parts)
        else:
            config_str = "—"

        # Show elapsed time: if finished, use finish_time; otherwise use current time
        if st.start_time:
            end_time = st.finish_time if st.finish_time else time.time()
            elapsed = fmt_duration(end_time - st.start_time)
        else:
            elapsed = "—"
        table.add_row(str(gpu_id), status, config_str, str(st.run_count), elapsed)

    parts = [Text(""), header, progress, Text(""), table]

    if errors:
        err_text = Text()
        for e in errors[-5:]:
            err_text.append(f"  • {e}\n", style="red")
        parts.append(Text(""))
        parts.append(Panel(err_text, title="[red]Errors[/red]", border_style="red"))

    footer_lines = []
    if sweep_url:
        footer_lines.append(f"  W&B  → {sweep_url}")
    if log_dir:
        footer_lines.append(f"  Logs → {log_dir}")
    if footer_lines:
        parts.append(Text("\n" + "\n".join(footer_lines), style="dim"))

    ts = datetime.now().strftime("%H:%M:%S")
    return Panel(
        Group(*parts),
        title=f"[bold blue]Hyperparameter Sweep[/bold blue] [dim]({ts})[/dim]",
        border_style="blue",
    )


def _build_eval_display(
    benchmarks_status: Dict[str, dict],
    total_tasks: int,
    done_count: int,
    elapsed_sec: float,
    errors: List[str],
    log_dir: Optional[Path] = None,
) -> "Panel":
    """Build a rich Panel showing eval progress and benchmark status."""
    header = Text("  ")
    header.append("Completed ", style="dim")
    header.append(str(done_count), style="bold green" if done_count > 0 else "bold")
    header.append(f"/{total_tasks}  ", style="dim")
    header.append("Elapsed ", style="dim")
    header.append(fmt_duration(elapsed_sec), style="bold")

    progress = _build_progress_bar(done_count, total_tasks)

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Benchmark", width=16)
    table.add_column("GPU", width=5, justify="center")
    table.add_column("Status", width=14)
    table.add_column("Elapsed", width=10, justify="right")

    status_styles = {
        "running": ("● Running", "yellow"),
        "done":    ("✓ Done", "green"),
        "failed":  ("✗ Failed", "red"),
        "queued":  ("◌ Queued", "dim"),
        "skipped": ("— Skipped", "dim"),
    }

    for bench in BENCHMARKS:
        info = benchmarks_status.get(bench, {})
        gpu = str(info.get("gpu", "—"))
        status_val = info.get("status", "queued")
        elapsed = info.get("elapsed", "—")

        s_text, s_style = status_styles.get(status_val, ("?", "dim"))
        status = Text(s_text, style=s_style)
        table.add_row(bench, gpu, status, elapsed)

    parts = [Text(""), header, progress, Text(""), table]

    if errors:
        err_text = Text()
        for e in errors[-5:]:
            err_text.append(f"  • {e}\n", style="red")
        parts.append(Text(""))
        parts.append(Panel(err_text, title="[red]Errors[/red]", border_style="red"))

    if log_dir:
        parts.append(Text(f"\n  Logs → {log_dir}", style="dim"))

    ts = datetime.now().strftime("%H:%M:%S")
    return Panel(
        Group(*parts),
        title=f"[bold blue]Benchmark Evaluation[/bold blue] [dim]({ts})[/dim]",
        border_style="blue",
    )


# =============================================================================
# Sweep Phase
# =============================================================================


def run_sweep_phase(
    dataset_path: str,
    paradigm: str,
    model: str,
    sweep_output_dir: Path,
    project: str,
    max_runs: int,
    sweep_method: Optional[str],
    sample_limit: Optional[int],
    seed: int,
    gpus: List[int],
    poll_interval: int = 30,
) -> str:
    """Run the sweep phase. Returns path to best model.

    Resumability:
    - best_model_info.json exists -> skip.
    - sweep_id.txt exists -> resume existing sweep.
    - Otherwise -> init new sweep.
    """
    best_info_path = sweep_output_dir / "best_model_info.json"
    sweep_id_path = sweep_output_dir / "sweep_id.txt"

    # Already completed?
    if best_info_path.exists():
        info = json.loads(best_info_path.read_text())
        log(f"[Sweep] Already completed. Best model: {info['path']} (loss={info['eval_loss']:.4f})")
        return info["path"]

    # Resolve sweep script
    if paradigm not in PARADIGM_SCRIPTS:
        print(f"Error: Unknown paradigm '{paradigm}'. Available: {', '.join(PARADIGM_SCRIPTS)}")
        sys.exit(1)
    sweep_script = ORCHESTRATOR_ROOT / PARADIGM_SCRIPTS[paradigm]
    if not sweep_script.exists():
        print(f"Error: Sweep script not found: {sweep_script}")
        sys.exit(1)

    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    # Init or resume sweep
    sweep_id = None
    if sweep_id_path.exists():
        sweep_id = sweep_id_path.read_text().strip()
        log(f"[Sweep] Resuming existing sweep: {sweep_id}")
        if sweep_method:
            log(f"[Sweep] NOTE: --sweep-method is ignored when resuming an existing sweep.")
        status = _get_sweep_status(sweep_id, project)
        log(f"[Sweep] Status: {status['state']} | Finished: {status['finished']} | "
            f"Running: {status['running']} | Failed: {status['failed']}")
        if status["state"] == "FINISHED":
            log("[Sweep] Sweep already finished. Resolving best model...")
            _write_best_model_from_wandb(sweep_id, project, sweep_output_dir)
            if best_info_path.exists():
                return json.loads(best_info_path.read_text())["path"]
            else:
                log("[Sweep] WARNING: Finished sweep has no usable results (all runs failed/incomplete)")
                log("[Sweep] Creating a fresh sweep instead...")
                # Archive the old sweep_id to avoid repeated checks
                if sweep_id_path.exists():
                    sweep_id_path.rename(sweep_id_path.with_suffix(".txt.old"))
                sweep_id = None  # Force new sweep creation below

    if sweep_id is None:
        log(f"[Sweep] Initializing new sweep: paradigm={paradigm}, model={model}")
        limit_str = f"{sample_limit:,}" if sample_limit else "all"
        log(f"[Sweep] Dataset: {dataset_path} | Max runs: {max_runs} | Samples: {limit_str}")

        cmd = [
            sys.executable, str(sweep_script),
            "--init",
            "--train-file", dataset_path,
            "--project", project,
            "--max-runs", str(max_runs),
            "--output-dir", str(sweep_output_dir),
            "--model", model,
        ]
        if sweep_method:
            cmd.extend(["--sweep-method", sweep_method])
        if sample_limit is not None:
            cmd.extend(["--sample-limit", str(sample_limit)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error initializing sweep:\n{result.stderr}")
            sys.exit(1)

        if not sweep_id_path.exists():
            print("Error: sweep_id.txt was not created by init.")
            sys.exit(1)
        sweep_id = sweep_id_path.read_text().strip()
        log(f"[Sweep] Created sweep: {sweep_id}")

    # Launch one agent per GPU, wait for all to finish
    _run_sweep_agents(sweep_id, project, sweep_script, sweep_output_dir,
                      max_runs, model, gpus, poll_interval)

    # Resolve best model from wandb (authoritative source)
    if not best_info_path.exists():
        if _RICH:
            _console.print("\n[dim]Querying W&B for best model...[/dim]")
        _write_best_model_from_wandb(sweep_id, project, sweep_output_dir)

    if best_info_path.exists():
        info = json.loads(best_info_path.read_text())

        if _RICH:
            best_info = Text()
            best_info.append("🏆 Best Model Found\n\n", style="bold green")
            best_info.append(f"Path: ", style="dim")
            best_info.append(f"{info['path']}\n", style="cyan")
            best_info.append(f"Eval Loss: ", style="dim")
            best_info.append(f"{info['eval_loss']:.4f}\n", style="bold yellow")

            config = info.get('config', {})
            if config:
                best_info.append(f"\nConfig:\n", style="dim")
                best_info.append(f"  LoRA Rank: {config.get('lora_rank', '?')}\n", style="white")
                best_info.append(f"  Learning Rate: {config.get('learning_rate', '?')}\n", style="white")
                best_info.append(f"  Epochs: {config.get('num_epochs', '?')}\n", style="white")

            _console.print(Panel(best_info, border_style="green"))
        else:
            log(f"[Sweep] Best model: {info['path']} (loss={info['eval_loss']:.4f})")

        return info["path"]
    else:
        log_dir = sweep_output_dir / "logs"
        print(f"Error: No best model found after sweep.")
        print(f"  This usually means ALL training runs failed or produced NaN loss.")
        print(f"  Check agent logs: {log_dir}")
        print(f"  Check wandb sweep: {sweep_id}")
        sys.exit(1)


def _run_sweep_agents(
    sweep_id: str,
    project: str,
    sweep_script: Path,
    output_dir: Path,
    max_runs: int,
    model: str,
    gpus: List[int],
    poll_interval: int,
):
    """Launch one sweep agent per GPU and wait until all exit."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Archive logs from any previous run so they aren't lost
    _archive_logs(log_dir)

    sweep_start = time.time()
    errors: List[str] = []

    # Resolve wandb sweep URL for dashboard
    sweep_url: Optional[str] = None
    try:
        import wandb
        entity = wandb.Api().default_entity
        sweep_url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
    except Exception:
        pass

    # Agent states for dashboard
    agent_states: Dict[int, _AgentState] = {}

    # Launch agents: gpu_id -> (proc, log_path)
    agents: Dict[int, Tuple[subprocess.Popen, Path]] = {}
    log(f"[Sweep] Launching {len(gpus)} agent(s) on GPUs {gpus}")
    for gpu_id in gpus:
        cmd = [
            sys.executable, "-u", str(sweep_script),
            "--sweep-id", sweep_id,
            "--project", project,
            "--output-dir", str(output_dir),
            "--model", model,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = log_dir / f"sweep_gpu{gpu_id}.log"
        label = f"sweep-gpu{gpu_id}"

        state = _AgentState(gpu_id=gpu_id, label=label, start_time=time.time())
        agent_states[gpu_id] = state

        proc = _launch_subprocess(cmd, env, log_path, label, agent_state=state)
        agents[gpu_id] = (proc, log_path)
        log(f"[Sweep]   GPU {gpu_id} -> PID {proc.pid}")

    # --- Poll until all agents exit ---
    sweep_status: Dict = {"finished": 0, "running": len(gpus), "failed": 0, "state": "RUNNING"}
    last_wandb_poll = 0.0

    def _poll_tick():
        """Check for finished processes and query wandb periodically."""
        nonlocal sweep_status, last_wandb_poll

        finished_gpus = []
        for gpu_id, (proc, log_path) in agents.items():
            if proc.poll() is not None:
                finished_gpus.append(gpu_id)

        for gpu_id in finished_gpus:
            proc, log_path = agents.pop(gpu_id)
            _untrack_process(proc)
            st = agent_states[gpu_id]
            st.finish_time = time.time()  # Store when agent finished
            if proc.returncode != 0:
                st.status = "failed"
                _report_failure(f"Sweep agent GPU {gpu_id}", proc.returncode, log_path, errors)
            else:
                # Only set to "done" if not already marked as "failed" by log parsing
                # (e.g., OOM errors that don't crash the agent process)
                if st.status != "failed":
                    st.status = "done"

        # Query wandb periodically (network call, avoid hammering)
        now = time.time()
        if now - last_wandb_poll >= poll_interval:
            sweep_status = _get_sweep_status(sweep_id, project)
            last_wandb_poll = now

    def _check_sweep_done_and_kill_agents():
        """Kill idle agents once all runs have actually completed (finished or failed).

        W&B sweep state "FINISHED" only means no new runs will be dispatched — it does
        NOT mean existing runs are done.  Killing on state==FINISHED would terminate
        agents that are actively training, causing all runs to crash.  We only kill
        when the completed+failed count shows every slot has been resolved.
        """
        finished = sweep_status.get("finished", 0)
        failed = sweep_status.get("failed", 0)
        if finished + failed >= max_runs:
            log(f"[Sweep] Sweep reached {finished + failed}/{max_runs} runs "
                f"(finished={finished}, failed={failed}). Stopping remaining agents...")
            for gpu_id, (proc, log_path) in list(agents.items()):
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                _untrack_process(proc)
                st = agent_states[gpu_id]
                if st.status not in ("done", "failed"):
                    st.status = "done"
                st.finish_time = time.time()
            agents.clear()

    if _RICH:
        with Live(
            _build_sweep_display(agent_states, sweep_status, max_runs, 0, errors, log_dir, sweep_url),
            console=_console,
            refresh_per_second=1,
        ) as live:
            while agents:
                time.sleep(3)
                _poll_tick()
                _check_sweep_done_and_kill_agents()
                elapsed = time.time() - sweep_start
                try:
                    live.update(_build_sweep_display(
                        agent_states, sweep_status, max_runs, elapsed, errors, log_dir, sweep_url))
                except Exception:
                    pass
                if not agents:
                    break
    else:
        # Fallback: original text-based output
        log(f"[Sweep] Fallback to text-based output")
        while agents:
            time.sleep(poll_interval)
            _poll_tick()
            _check_sweep_done_and_kill_agents()
            elapsed = time.time() - sweep_start
            log(f"[Sweep] Runs: {sweep_status['finished']}/{max_runs} done, "
                f"{sweep_status['running']} running"
                + (f", {sweep_status['failed']} failed" if sweep_status['failed'] else "")
                + f" | Agents alive: {len(agents)} | Elapsed: {fmt_duration(elapsed)}")
            if not agents:
                break

    elapsed = time.time() - sweep_start

    # Show completion status
    if _RICH:
        completion_text = Text()
        completion_text.append("✅ All agents finished in ", style="green")
        completion_text.append(fmt_duration(elapsed), style="bold green")

        if errors:
            completion_text.append("\n\n⚠️  Errors encountered:\n", style="yellow")
            for err in errors[-3:]:  # Show last 3 errors
                completion_text.append(f"  • {err}\n", style="red")

        _console.print(Panel(
            completion_text,
            title="[bold blue]Sweep Phase Complete[/bold blue]",
            border_style="blue"
        ))
    else:
        log(f"[Sweep] All agents finished in {fmt_duration(elapsed)}")
        if errors:
            for err in errors:
                log(f"  ERROR: {err}")


def _get_sweep_status(sweep_id: str, project: str) -> Dict:
    """Query wandb for sweep status."""
    try:
        import wandb
        api = wandb.Api()
        entity = api.default_entity
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        return {
            "state": sweep.state,
            "finished": sum(1 for r in sweep.runs if r.state == "finished"),
            "running": sum(1 for r in sweep.runs if r.state == "running"),
            "failed": sum(1 for r in sweep.runs if r.state in ("failed", "crashed")),
        }
    except Exception as e:
        log(f"[Sweep] Warning: could not query wandb: {e}")
        return {"state": "unknown", "finished": 0, "running": 0, "failed": 0}


def _write_best_model_from_wandb(sweep_id: str, project: str, output_dir: Path):
    """Query wandb for the best run and write best_model_info.json.

    Matches the best run's config to the correct model directory on disk.
    """
    try:
        import wandb
        api = wandb.Api()
        entity = api.default_entity
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

        # Find best run by eval loss
        import math
        best_loss = float("inf")
        best_run = None
        total_runs = 0
        finished_runs = 0
        runs_with_loss = 0
        for run in sweep.runs:
            total_runs += 1
            if run.state != "finished":
                continue
            finished_runs += 1
            # Check all possible metric names (HF Trainer may log as eval_loss with underscore)
            loss = (run.summary.get("eval/loss")
                    or run.summary.get("final/eval_loss")
                    or run.summary.get("eval_loss"))
            if loss is not None and not math.isnan(loss) and loss < best_loss:
                runs_with_loss += 1
                best_loss = loss
                best_run = run

        if best_run is None:
            log(f"[Sweep] No usable runs found in wandb. "
                f"Total: {total_runs}, Finished: {finished_runs}, With valid loss: {runs_with_loss}")
            # Log per-run details for debugging
            for run in sweep.runs:
                loss = (run.summary.get("eval/loss")
                        or run.summary.get("final/eval_loss")
                        or run.summary.get("eval_loss"))
                error = run.summary.get("error", "")
                log(f"  Run {run.id}: state={run.state}, loss={loss}, error={error}")
            return

        config = {
            "lora_rank": best_run.config.get("lora_rank"),
            "learning_rate": best_run.config.get("learning_rate"),
            "num_epochs": best_run.config.get("num_epochs"),
            "eval_loss": best_loss,
        }

        # Find the matching model directory on disk.
        # The sweep script saves best models under output_dir/best/<descriptive-name>/
        # with the loss encoded in the name. Pick the one with the lowest loss.
        best_dir = output_dir / "best"
        if not best_dir.exists():
            log("[Sweep] No best/ directory found on disk.")
            return

        candidates = [d for d in best_dir.iterdir() if d.is_dir()]
        if not candidates:
            log("[Sweep] No model directories found under best/.")
            return

        # Parse loss from directory names like "best-r64-lr1e-4-e2-loss0_1234"
        # and pick the directory with the lowest loss
        best_path = None
        best_parsed_loss = float("inf")
        for d in candidates:
            try:
                # Extract the loss part: everything after "loss"
                name = d.name
                loss_part = name.split("-loss")[-1]
                parsed_loss = float(loss_part.replace("_", "."))
                if parsed_loss < best_parsed_loss:
                    best_parsed_loss = parsed_loss
                    best_path = str(d.absolute())
            except (ValueError, IndexError):
                # Can't parse loss from name, use as fallback
                if best_path is None:
                    best_path = str(d.absolute())

        if best_path is None:
            return

        info = {
            "path": best_path,
            "config": config,
            "eval_loss": best_loss,
            "run_id": best_run.id,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "best_model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        log(f"[Sweep] Wrote best_model_info.json (loss={best_loss:.4f})")

    except Exception as e:
        import traceback
        log(f"[Sweep] Warning: could not write best model info: {e}")
        log(f"[Sweep] Traceback:\n{traceback.format_exc()}")


# =============================================================================
# Eval Phase
# =============================================================================


def _find_results_count(bench_dir: Path, bench: str) -> int:
    """Count existing results for a benchmark."""
    direct = bench_dir / f"{bench}_results.jsonl"
    if direct.exists():
        return count_jsonl_lines(direct)
    for f in bench_dir.glob(f"*/{bench}_results.jsonl"):
        return count_jsonl_lines(f)
    return 0


def run_eval_phase(
    best_model_path: str,
    eval_output_dir: Path,
    eval_limit: Optional[int],
    seed: int,
    save_interval: int,
    gpus: List[int],
    poll_interval: int = 15,
):
    """Run all benchmarks on the best model, distributing across GPUs.

    If len(benchmarks) > len(gpus), waits for a GPU to free up before
    launching the next benchmark.
    """
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    run_eval_script = ORCHESTRATOR_ROOT / "evals" / "run_eval.py"

    # Build task queue, skipping already-completed benchmarks
    task_queue: List[Tuple[str, Path]] = []
    benchmarks_status: Dict[str, dict] = {}

    for bench in BENCHMARKS:
        bench_dir = eval_output_dir / bench
        completed = _find_results_count(bench_dir, bench)

        if eval_limit and completed >= eval_limit:
            log(f"[Eval] {bench}: already completed ({completed}/{eval_limit}). Skipping.")
            benchmarks_status[bench] = {"status": "skipped", "gpu": "—", "elapsed": "—"}
            continue
        elif completed > 0:
            log(f"[Eval] {bench}: found {completed} existing results, will auto-resume")
        else:
            log(f"[Eval] {bench}: starting fresh" + (f" (limit: {eval_limit})" if eval_limit else ""))
        task_queue.append((bench, bench_dir))
        benchmarks_status[bench] = {"status": "queued", "gpu": "—", "elapsed": "—"}

    if not task_queue:
        log("[Eval] All benchmarks already completed!")
        return

    total_tasks = len(task_queue)
    log(f"[Eval] {total_tasks} benchmark(s) to run: {[t[0] for t in task_queue]}")
    log(f"[Eval] Using {len(gpus)} GPU(s): {gpus}")

    log_dir = eval_output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Archive logs from any previous eval run
    _archive_logs(log_dir)

    # active: gpu_id -> (bench_name, proc, log_path, start_time)
    active: Dict[int, Tuple[str, subprocess.Popen, Path, float]] = {}
    available_gpus = list(gpus)  # GPUs not currently running a benchmark
    completed_benchmarks = []
    failed_benchmarks = []
    errors: List[str] = []
    eval_start = time.time()

    def _launch_eval(bench: str, bench_dir: Path, gpu_id: int):
        cmd = [
            sys.executable, "-u", str(run_eval_script),
            "--benchmark", bench,
            "--model", best_model_path,
            "--output-dir", str(bench_dir),
            "--save-interval", str(save_interval),
            "--seed", str(seed),
            "-vv",
        ]
        if eval_limit:
            cmd.extend(["--limit", str(eval_limit)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = log_dir / f"eval_{bench}_gpu{gpu_id}.log"
        label = f"eval-{bench}-gpu{gpu_id}"
        proc = _launch_subprocess(cmd, env, log_path, label)
        active[gpu_id] = (bench, proc, log_path, time.time())
        benchmarks_status[bench] = {
            "status": "running", "gpu": str(gpu_id), "elapsed": "—",
        }
        log(f"[Eval] Launched {bench} on GPU {gpu_id} (PID {proc.pid})")

    # Initial launch: fill all available GPUs
    while task_queue and available_gpus:
        gpu_id = available_gpus.pop(0)
        bench, bench_dir = task_queue.pop(0)
        _launch_eval(bench, bench_dir, gpu_id)

    def _eval_poll_tick():
        """Check for finished evals and launch queued tasks on freed GPUs."""
        freed_gpus = []
        for gpu_id, (bench, proc, log_path, start) in list(active.items()):
            # Update elapsed time for running benchmarks
            benchmarks_status[bench]["elapsed"] = fmt_duration(time.time() - start)

            if proc.poll() is not None:
                elapsed_bench = time.time() - start
                _untrack_process(proc)
                if proc.returncode != 0:
                    _report_failure(f"Eval {bench}", proc.returncode, log_path, errors)
                    failed_benchmarks.append(bench)
                    benchmarks_status[bench]["status"] = "failed"
                    benchmarks_status[bench]["elapsed"] = fmt_duration(elapsed_bench)
                else:
                    completed_benchmarks.append(bench)
                    benchmarks_status[bench]["status"] = "done"
                    benchmarks_status[bench]["elapsed"] = fmt_duration(elapsed_bench)
                del active[gpu_id]
                freed_gpus.append(gpu_id)

        # Launch queued tasks on freed GPUs
        for gpu_id in freed_gpus:
            if task_queue:
                bench, bench_dir = task_queue.pop(0)
                _launch_eval(bench, bench_dir, gpu_id)

    # --- Monitor loop ---
    if _RICH:
        done_count = len(completed_benchmarks) + len(failed_benchmarks)
        with Live(
            _build_eval_display(benchmarks_status, total_tasks, done_count, 0, errors, log_dir),
            console=_console,
            refresh_per_second=1,
        ) as live:
            while active:
                time.sleep(3)
                _eval_poll_tick()
                elapsed = time.time() - eval_start
                done_count = len(completed_benchmarks) + len(failed_benchmarks)
                try:
                    live.update(_build_eval_display(
                        benchmarks_status, total_tasks, done_count, elapsed, errors, log_dir))
                except Exception:
                    pass
    else:
        # Fallback: original text-based output
        while active:
            time.sleep(poll_interval)
            _eval_poll_tick()
            elapsed = time.time() - eval_start
            done = len(completed_benchmarks) + len(failed_benchmarks)
            active_names = [v[0] for v in active.values()]
            log(f"[Eval] Done: {done}/{total_tasks} | Active: {active_names} | "
                f"Queue: {len(task_queue)} | Elapsed: {fmt_duration(elapsed)}")

    # Report
    if failed_benchmarks:
        if _RICH:
            fail_text = Text()
            fail_text.append(f"⚠️  {len(failed_benchmarks)} benchmark(s) failed: ", style="bold yellow")
            fail_text.append(f"{', '.join(failed_benchmarks)}\n\n", style="red")
            if errors:
                fail_text.append("Errors:\n", style="dim")
                for err in errors[-3:]:
                    fail_text.append(f"  • {err}\n", style="red")
            fail_text.append("\n💡 Re-run the pipeline to retry failed benchmarks.", style="dim")

            _console.print(Panel(fail_text, title="[yellow]Evaluation Warnings[/yellow]", border_style="yellow"))
        else:
            log(f"[Eval] WARNING: {len(failed_benchmarks)} benchmark(s) failed: {failed_benchmarks}")
            if errors:
                for err in errors:
                    log(f"  ERROR: {err}")
            log("[Eval] Re-run the pipeline to retry failed benchmarks.")


# =============================================================================
# Summary
# =============================================================================


def print_summary(results_dir: Path):
    print()
    print("=" * 70)
    print("PIPELINE COMPLETE - RESULTS SUMMARY")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print()

    best_info_path = results_dir / "sweep" / "best_model_info.json"
    if best_info_path.exists():
        info = json.loads(best_info_path.read_text())
        print("SWEEP (Best Model):")
        print(f"  Path:       {info['path']}")
        print(f"  Eval Loss:  {info['eval_loss']:.4f}")
        cfg = info.get("config", {})
        if cfg:
            print(f"  LoRA Rank:  {cfg.get('lora_rank', '?')}")
            print(f"  LR:         {cfg.get('learning_rate', '?')}")
            print(f"  Epochs:     {cfg.get('num_epochs', '?')}")
        print()

    eval_dir = results_dir / "eval"
    if eval_dir.exists():
        print("EVALUATION BENCHMARKS:")
        for bench in BENCHMARKS:
            metrics_file = eval_dir / bench / f"{bench}_metrics.json"
            results_file = eval_dir / bench / f"{bench}_results.jsonl"
            rows = count_jsonl_lines(results_file)
            if metrics_file.exists():
                metrics = json.loads(metrics_file.read_text())
                if bench == "hle":
                    print(f"  HLE:          accuracy={metrics.get('accuracy', 0):.2%}  ({rows} questions)")
                elif bench == "gaia":
                    print(f"  GAIA:         accuracy={metrics.get('accuracy', 0):.2%}  ({rows} tasks)")
                elif bench == "simpleqa":
                    print(f"  SimpleQA:     f1={metrics.get('f1', 0):.2%}  ({rows} questions)")
                elif bench == "deepresearch":
                    print(f"  DeepResearch: score={metrics.get('overall_score', 0):.4f}  ({rows} tasks)")
            elif rows > 0:
                print(f"  {bench}: {rows} results (metrics not yet computed)")
            else:
                print(f"  {bench}: not started")
        print()

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Training Pipeline: Sweep -> Best Checkpoint -> Eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline on 8 GPUs
    python training/pipeline.py \\
        --dataset ./data/filtered_traces.parquet \\
        --model Qwen/Qwen3-8B \\
        --paradigm LORA --max-runs 12 \\
        --gpus 0,1,2,3,4,5,6,7

    # Resume a partially completed pipeline
    python training/pipeline.py \\
        --dataset ./data/filtered_traces.parquet \\
        --model Qwen/Qwen3-8B \\
        --gpus 0,1,2,3
        """,
    )

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset file (parquet, json, jsonl)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model name or HuggingFace path")
    parser.add_argument("--paradigm", type=str, default="LORA",
                        choices=list(PARADIGM_SCRIPTS.keys()),
                        help="Training paradigm (default: LORA)")
    parser.add_argument("--project", type=str, default="orchestrator-lora-sweep",
                        help="W&B project name")
    parser.add_argument("--max-runs", type=int, default=12,
                        help="Maximum sweep runs (default: 12)")
    parser.add_argument("--sweep-method", type=str, default="bayes",
                        choices=["bayes", "grid", "random"],
                        help="W&B sweep method (default: use sweep script default; options: bayes, grid, random)")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Limit training samples per sweep run (default: use full dataset)")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Max samples per benchmark (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save eval results every N minutes (default: 10)")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU indices to use (e.g. 0,1,2,3)")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Only run the sweep phase, skip eval")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run eval phase (requires existing best model)")

    args = parser.parse_args()

    # Parse GPU list
    try:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
    except ValueError:
        print(f"Error: Invalid --gpus format: '{args.gpus}'. Expected comma-separated ints (e.g. 0,1,2,3)")
        sys.exit(1)
    if not gpus:
        print("Error: --gpus must specify at least one GPU.")
        sys.exit(1)

    # Validate dataset
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    # Results directory
    results_dir = dataset_path.parent / "results"
    sweep_output_dir = results_dir / "sweep"
    eval_output_dir = results_dir / "eval"

    global _sweep_output_dir
    _sweep_output_dir = sweep_output_dir

    # Banner
    print()
    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Dataset:    {dataset_path}")
    print(f"  Model:      {args.model}")
    print(f"  Paradigm:   {args.paradigm}")
    print(f"  Project:    {args.project}")
    print(f"  Max Runs:   {args.max_runs}")
    print(f"  GPUs:       {gpus}")
    print(f"  Seed:       {args.seed}")
    print(f"  Results:    {results_dir}")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    pipeline_start = time.time()

    # ---- Sweep Phase ----
    if not args.eval_only:
        log("=" * 50)
        log("PHASE 1: HYPERPARAMETER SWEEP")
        log("=" * 50)

        best_model_path = run_sweep_phase(
            dataset_path=str(dataset_path),
            paradigm=args.paradigm,
            model=args.model,
            sweep_output_dir=sweep_output_dir,
            project=args.project,
            max_runs=args.max_runs,
            sweep_method=args.sweep_method,
            sample_limit=args.sample_limit,
            seed=args.seed,
            gpus=gpus,
        )

        sweep_elapsed = time.time() - pipeline_start

        if _RICH:
            transition_text = Text()
            transition_text.append("⏱️  Sweep completed in ", style="dim")
            transition_text.append(fmt_duration(sweep_elapsed), style="bold cyan")
            transition_text.append("\n\n🚀 Starting benchmark evaluation...\n", style="green")
            transition_text.append(f"Model: ", style="dim")
            transition_text.append(f"{best_model_path}\n", style="cyan")
            transition_text.append(f"\nBenchmarks: ", style="dim")
            transition_text.append("HLE, GAIA, SimpleQA, DeepResearch", style="yellow")

            _console.print(Panel(
                transition_text,
                title="[bold blue]Phase 1 → Phase 2[/bold blue]",
                border_style="blue"
            ))
            time.sleep(1.5)  # Let user see the transition
        else:
            log(f"[Sweep] Phase complete in {fmt_duration(sweep_elapsed)}")
            log(f"[Sweep] Best model: {best_model_path}")
            print()
    else:
        best_info_path = sweep_output_dir / "best_model_info.json"
        if not best_info_path.exists():
            print("Error: --eval-only requires a completed sweep. No best_model_info.json found.")
            sys.exit(1)
        info = json.loads(best_info_path.read_text())
        best_model_path = info["path"]

        if _RICH:
            _console.print(f"[dim]Using best model:[/dim] [cyan]{best_model_path}[/cyan]")
        else:
            log(f"[Eval-only] Using best model: {best_model_path}")

    if args.sweep_only:
        if _RICH:
            _console.print("\n[yellow]--sweep-only: Skipping eval phase.[/yellow]\n")
        else:
            log("--sweep-only: Skipping eval phase.")
        print_summary(results_dir)
        return

    # ---- Eval Phase ----
    eval_start = time.time()

    if not _RICH:
        log("=" * 50)
        log("PHASE 2: BENCHMARK EVALUATION")
        log("=" * 50)

    run_eval_phase(
        best_model_path=best_model_path,
        eval_output_dir=eval_output_dir,
        eval_limit=args.eval_limit,
        seed=args.seed,
        save_interval=args.save_interval,
        gpus=gpus,
    )

    log(f"[Eval] Phase complete in {fmt_duration(time.time() - eval_start)}")
    log(f"[Pipeline] Total time: {fmt_duration(time.time() - pipeline_start)}")
    print_summary(results_dir)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup_all_processes)
    main()
