#!/usr/bin/env python3
"""Build paper + appendix figures/tables from H100 experiment artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns


DATE_RE = re.compile(r"^(\d{4}-\d{2})_(.+?)__(.+)$")

MODEL_LABELS = {
    "M0": "M0 (SFT)",
    "M1": "M1 (Throughput GRPO)",
    "M2": "M2 (Energy GRPO)",
    "M3": "M3 (Energy+IPW GRPO)",
    "M4": "M4 (Runtime PPO)",
    "M5": "M5 (HRL)",
}


def _parse_folder(folder: str) -> Tuple[str, str, str]:
    m = DATE_RE.match(folder)
    if not m:
        return ("", "", folder)
    return m.group(1), m.group(2), m.group(3)


def _model_short(model_tag: str) -> str:
    if model_tag.startswith("M0"):
        return "M0"
    if model_tag.startswith("M1"):
        return "M1"
    if model_tag.startswith("M2"):
        return "M2"
    if model_tag.startswith("M3"):
        return "M3"
    if model_tag.startswith("M4"):
        return "M4"
    if model_tag.startswith("M5"):
        return "M5"
    return model_tag


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_experiments(root: Path) -> Dict[str, dict]:
    exps: Dict[str, dict] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "organized_figures":
            continue
        month, model_tag, exp_tag = _parse_folder(d.name)
        if not month:
            continue

        stem = d.name
        summary_path = d / f"{stem}_summary.json"
        metrics_path = d / f"{stem}_metrics.csv"
        per_seed_path = d / f"{stem}_per_seed.csv"
        per_task_path = d / f"{stem}_per_task.jsonl"
        failure_path = d / f"{stem}_failure_taxonomy.csv"

        summary = _safe_read_json(summary_path) or {}
        metrics = _safe_read_csv(metrics_path)
        per_seed = _safe_read_csv(per_seed_path)
        per_task = None
        if per_task_path.exists():
            try:
                per_task = pd.read_json(per_task_path, lines=True)
            except Exception:
                per_task = None
        failure = _safe_read_csv(failure_path)

        if metrics is not None:
            metrics = metrics.copy()
            metrics["folder"] = stem
            metrics["model_tag"] = model_tag
            metrics["model_short"] = _model_short(model_tag)
            metrics["experiment_tag"] = exp_tag
            if "task_id" in metrics.columns:
                parsed = metrics["task_id"].astype(str).str.extract(r"^L(\d+)_(\d+)$")
                metrics["task_level"] = pd.to_numeric(parsed[0], errors="coerce")
                metrics["task_index"] = pd.to_numeric(parsed[1], errors="coerce")
                # Keep task_num for compatibility with existing plotting logic.
                metrics["task_num"] = metrics["task_index"].fillna(0).astype(int)
            if "turns_to_success" not in metrics.columns:
                if per_task is not None and {"run_id", "turns_to_success"}.issubset(set(per_task.columns)):
                    metrics = metrics.merge(per_task[["run_id", "turns_to_success"]], on="run_id", how="left")
                    metrics["turns_to_success"] = metrics["turns_to_success"].fillna(
                        np.where(metrics.get("correct", 0).astype(float) > 0, 1, -1)
                    )
                else:
                    metrics["turns_to_success"] = np.where(metrics.get("correct", 0).astype(float) > 0, 1, -1)

        exps[stem] = {
            "path": d,
            "month": month,
            "model_tag": model_tag,
            "model_short": _model_short(model_tag),
            "experiment_tag": exp_tag,
            "summary": summary,
            "metrics": metrics,
            "per_seed": per_seed,
            "per_task": per_task,
            "failure": failure,
        }
    return exps


def get_core_experiment_names(exps: Dict[str, dict]) -> Dict[str, str]:
    names = {}
    for name in exps:
        if "__kernel_generation_baseline" in name:
            names["M0"] = name
        elif "__throughput_rl" in name and "_M1_" in name:
            names["M1"] = name
        elif "__energy_aware_rl" in name and "_M2_" in name:
            names["M2"] = name
        elif "__ipw_blend_sweep" in name and "_M3_" in name:
            names["M3"] = name
        elif "__runtime_control" in name and "_M4_" in name:
            names["M4"] = name
        elif "__hierarchical_control" in name and "_M5_" in name:
            names["M5"] = name
    if "M3" not in names:
        for name in exps:
            if "__ipw_blend_lambda_ablation" in name and "_M3_" in name:
                names["M3"] = name
                break
    return names


def ensure_dirs(out: Path) -> Dict[str, Path]:
    dirs = {
        "root": out,
        "main_figures": out / "main_figures",
        "appendix_figures": out / "appendix_figures",
        "tables": out / "tables",
        "data": out / "data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _dominant_points(df: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    vals = df[[x_col, y_col]].to_numpy()
    dominated = np.zeros(len(vals), dtype=bool)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            if vals[j, 0] <= vals[i, 0] and vals[j, 1] <= vals[i, 1] and (
                vals[j, 0] < vals[i, 0] or vals[j, 1] < vals[i, 1]
            ):
                dominated[i] = True
                break
    return ~dominated


def plot_main_figures(exps: Dict[str, dict], core: Dict[str, str], dirs: Dict[str, Path]) -> List[str]:
    made: List[str] = []

    # 1) Accuracy-Energy Pareto Frontier with CI ellipses
    rows = []
    for m in ["M0", "M1", "M2", "M3"]:
        exp_name = core.get(m)
        if not exp_name:
            continue
        df = exps[exp_name]["metrics"]
        if df is None or df.empty:
            continue
        seed_df = (
            df.groupby("seed", as_index=False)
            .agg(runtime_ms=("runtime_ms", "mean"), joules=("joules", "mean"), correctness=("correct", "mean"))
            .assign(model=m)
        )
        rows.append(seed_df)
    if rows:
        p = pd.concat(rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(8.5, 6))
        palette = {"M0": "#4C78A8", "M1": "#F58518", "M2": "#54A24B", "M3": "#E45756"}
        centers = []
        for model, g in p.groupby("model"):
            c = palette.get(model, "#333333")
            ax.scatter(g["runtime_ms"], g["joules"], color=c, alpha=0.55, s=40)
            mx, my = g["runtime_ms"].mean(), g["joules"].mean()
            sx = g["runtime_ms"].std(ddof=1) if len(g) > 1 else 0.0
            sy = g["joules"].std(ddof=1) if len(g) > 1 else 0.0
            ell = Ellipse((mx, my), width=2 * 1.96 * sx, height=2 * 1.96 * sy, facecolor=c, alpha=0.15, edgecolor=c)
            ax.add_patch(ell)
            ax.scatter([mx], [my], color=c, s=100, edgecolor="black", linewidth=0.8, label=MODEL_LABELS.get(model, model))
            centers.append({"model": model, "runtime_ms": mx, "joules": my})
        cdf = pd.DataFrame(centers)
        if not cdf.empty:
            nd_mask = _dominant_points(cdf, "runtime_ms", "joules")
            nd = cdf[nd_mask]
            ax.scatter(nd["runtime_ms"], nd["joules"], s=170, facecolors="none", edgecolors="black", linewidths=1.6)
        ax.set_title("Accuracy-Energy Pareto Frontier (Seed Means + 95% CI Ellipses)")
        ax.set_xlabel("Runtime (ms, lower is better)")
        ax.set_ylabel("Energy (Joules, lower is better)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        out = dirs["main_figures"] / "01_accuracy_energy_pareto_frontier.png"
        _savefig(out)
        made.append(str(out))

    # 2) Pass@k vs Turns Curve
    mapping = {
        "single_shot": None,
        "multiturn": None,
        "rl_initialized_multiturn": None,
    }
    for name in exps:
        if "__single_shot_generation" in name and "_M0_" in name:
            mapping["single_shot"] = name
        elif "__multiturn_generation" in name and "_M0_" in name:
            mapping["multiturn"] = name
        elif "__single_shot_vs_multiturn" in name:
            mapping["rl_initialized_multiturn"] = name

    curves = []
    for label, exp_name in mapping.items():
        if not exp_name:
            continue
        df = exps[exp_name]["metrics"]
        if df is None or df.empty or "turns_to_success" not in df.columns:
            continue
        tmp = df.copy()
        tmp["turns_to_success"] = pd.to_numeric(tmp["turns_to_success"], errors="coerce").fillna(-1).astype(int)
        n_total = tmp.groupby("seed")["task_id"].nunique()
        for t in [1, 2, 3, 4, 5]:
            solved = tmp[(tmp["turns_to_success"] > 0) & (tmp["turns_to_success"] <= t)].groupby("seed")["task_id"].nunique()
            passk = (solved / n_total).fillna(0.0)
            for seed, val in passk.items():
                curves.append({"curve": label, "seed": seed, "turn": t, "passk": float(val)})
    if curves:
        cdf = pd.DataFrame(curves)
        g = cdf.groupby(["curve", "turn"], as_index=False).agg(mean=("passk", "mean"), std=("passk", "std"))
        fig, ax = plt.subplots(figsize=(8, 5))
        for curve_name, gg in g.groupby("curve"):
            gg = gg.sort_values("turn")
            std = gg["std"].fillna(0.0)
            ax.plot(gg["turn"], gg["mean"], marker="o", label=curve_name.replace("_", " "))
            ax.fill_between(gg["turn"], gg["mean"] - 1.96 * std, gg["mean"] + 1.96 * std, alpha=0.15)
        ax.set_title("Pass@k vs Turns")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Cumulative Pass@k")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        out = dirs["main_figures"] / "02_passk_vs_turns_curve.png"
        _savefig(out)
        made.append(str(out))

    # 3) Matched-runtime energy advantage (M3 vs M1)
    if core.get("M1") and core.get("M3"):
        m1 = exps[core["M1"]]["metrics"]
        m3 = exps[core["M3"]]["metrics"]
        if m1 is not None and m3 is not None:
            pair = (
                m1.merge(m3, on=["seed", "task_id"], suffixes=("_m1", "_m3"))
                .assign(runtime_diff_pct=lambda d: ((d["runtime_ms_m3"] - d["runtime_ms_m1"]) / d["runtime_ms_m1"]).abs() * 100.0)
                .assign(delta_joules_pct=lambda d: (d["joules_m3"] - d["joules_m1"]) / d["joules_m1"] * 100.0)
            )
            pair = pair[pair["runtime_diff_pct"] <= 3.0].copy()
            if not pair.empty:
                pair["pair_id"] = pair["task_id"] + "_s" + pair["seed"].astype(str)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
                sns.stripplot(data=pair, x="task_id", y="delta_joules_pct", ax=axes[0], size=4, alpha=0.6)
                axes[0].axhline(0, color="black", linewidth=1)
                axes[0].set_title("Matched Runtime (|delta runtime| <= 3%): delta Joules% (M3 - M1)")
                axes[0].set_xlabel("Task")
                axes[0].set_ylabel("delta Joules (%)")
                axes[0].tick_params(axis="x", rotation=90)
                sns.boxplot(data=pair, y="delta_joules_pct", ax=axes[1], color="#9ecae1")
                sns.stripplot(data=pair, y="delta_joules_pct", ax=axes[1], color="#1f77b4", alpha=0.5)
                axes[1].axhline(0, color="black", linewidth=1)
                axes[1].set_title("Distribution")
                axes[1].set_xlabel("")
                out = dirs["main_figures"] / "03_matched_runtime_energy_advantage.png"
                _savefig(out)
                made.append(str(out))

                pair[["seed", "task_id", "runtime_diff_pct", "delta_joules_pct"]].to_csv(
                    dirs["data"] / "03_matched_runtime_pairs.csv", index=False
                )

    # 4) Reward-to-outcome decomposition waterfall (M0->M1->M2->M3)
    if all(k in core for k in ["M0", "M1", "M2", "M3"]):
        s0 = exps[core["M0"]]["summary"]
        s1 = exps[core["M1"]]["summary"]
        s2 = exps[core["M2"]]["summary"]
        s3 = exps[core["M3"]]["summary"]
        r0 = float(s0.get("reward_mean", 0.0))
        r1 = float(s1.get("reward_mean", 0.0))
        r2 = float(s2.get("reward_mean", 0.0))
        r3 = float(s3.get("reward_mean", 0.0))
        deltas = [r0, r1 - r0, r2 - r1, r3 - r2]
        labels = ["Base (M0)", "+ Throughput (M1)", "+ Energy (M2)", "+ IPW Blend (M3)"]
        cumulative = np.cumsum(deltas)
        starts = np.r_[0, cumulative[:-1]]
        colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (lab, d, st, c) in enumerate(zip(labels, deltas, starts, colors)):
            ax.bar(i, d, bottom=st, color=c, width=0.6)
            ax.text(i, st + d + (0.03 if d >= 0 else -0.03), f"{d:+.2f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel("Reward Mean")
        ax.set_title("Reward-to-Outcome Decomposition (Waterfall)")
        out = dirs["main_figures"] / "04_reward_outcome_decomposition_waterfall.png"
        _savefig(out)
        made.append(str(out))

    # 5) Failure taxonomy transition (proxy from M0 -> M3)
    if core.get("M0") and core.get("M3"):
        f0 = exps[core["M0"]]["failure"]
        f3 = exps[core["M3"]]["failure"]
        if f0 is not None and f3 is not None and {"reason", "count"}.issubset(f0.columns) and {"reason", "count"}.issubset(
            f3.columns
        ):
            a = f0.groupby("reason", as_index=False)["count"].sum().rename(columns={"count": "early"})
            b = f3.groupby("reason", as_index=False)["count"].sum().rename(columns={"count": "late"})
            ff = a.merge(b, on="reason", how="outer").fillna(0.0)
            ff["delta"] = ff["late"] - ff["early"]
            ff = ff.sort_values("early", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(ff))
            w = 0.38
            ax.bar(x - w / 2, ff["early"], width=w, label="Early (M0 baseline)", color="#9ecae1")
            ax.bar(x + w / 2, ff["late"], width=w, label="Later (M3 IPW)", color="#31a354")
            ax.set_xticks(x)
            ax.set_xticklabels(ff["reason"], rotation=40, ha="right")
            ax.set_ylabel("Count")
            ax.set_title("Failure Taxonomy Shift (Early -> Later)")
            ax.legend(frameon=False)
            out = dirs["main_figures"] / "05_failure_taxonomy_transition.png"
            _savefig(out)
            made.append(str(out))

    # 6) Cross-hardware transfer scatter (proxy: final_eval vs transfer split)
    final_name = next((n for n in exps if "__final_eval_suite" in n and "_M_ALL_" in n), None)
    transfer_name = next((n for n in exps if "__cross_hardware_transfer" in n and "_M_ALL_" in n), None)
    if final_name and transfer_name:
        a = exps[final_name]["metrics"]
        b = exps[transfer_name]["metrics"]
        if a is not None and b is not None:
            a_t = a.groupby("task_id", as_index=False).agg(runtime_a=("runtime_ms", "mean"), joules_a=("joules", "mean"))
            b_t = b.groupby("task_id", as_index=False).agg(runtime_b=("runtime_ms", "mean"), joules_b=("joules", "mean"))
            m = a_t.merge(b_t, on="task_id", how="inner")
            if not m.empty:
                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                axes[0].scatter(m["runtime_a"], m["runtime_b"], alpha=0.7)
                mn = min(m["runtime_a"].min(), m["runtime_b"].min())
                mx = max(m["runtime_a"].max(), m["runtime_b"].max())
                axes[0].plot([mn, mx], [mn, mx], "--", color="gray")
                axes[0].set_xlabel("Reference Runtime (ms)")
                axes[0].set_ylabel("Transfer Runtime (ms)")
                axes[0].set_title("Runtime Transfer")

                axes[1].scatter(m["joules_a"], m["joules_b"], alpha=0.7, color="#2ca02c")
                mn = min(m["joules_a"].min(), m["joules_b"].min())
                mx = max(m["joules_a"].max(), m["joules_b"].max())
                axes[1].plot([mn, mx], [mn, mx], "--", color="gray")
                axes[1].set_xlabel("Reference Joules")
                axes[1].set_ylabel("Transfer Joules")
                axes[1].set_title("Energy Transfer")
                fig.suptitle("Cross-Hardware Transfer Scatter (Reference vs Transfer Split)")
                out = dirs["main_figures"] / "06_cross_hardware_transfer_scatter.png"
                _savefig(out)
                made.append(str(out))

    # 7) Domain coverage stacked bar
    if core.get("M1"):
        baseline = exps[core["M1"]]["metrics"]
        threshold = float(baseline[baseline["correct"] == 1]["joules"].median()) if baseline is not None else np.nan
        cover_rows = []
        for m in ["M0", "M1", "M2", "M3", "M4", "M5"]:
            if m not in core:
                continue
            df = exps[core[m]]["metrics"]
            if df is None or df.empty:
                continue
            grp = df.groupby("task_id", as_index=False).agg(correct=("correct", "mean"), joules=("joules", "mean"))
            solved = grp["correct"] >= 0.5
            energy_eff = solved & (grp["joules"] <= threshold)
            cover_rows.append(
                {
                    "model": m,
                    "energy_efficient_solved": int(energy_eff.sum()),
                    "solved_not_energy_efficient": int((solved & ~energy_eff).sum()),
                    "unsolved_or_fallback": int((~solved).sum()),
                }
            )
        if cover_rows:
            cdf = pd.DataFrame(cover_rows).set_index("model")
            fig, ax = plt.subplots(figsize=(9, 5))
            cdf = cdf.loc[[m for m in ["M0", "M1", "M2", "M3", "M4", "M5"] if m in cdf.index]]
            bottom = np.zeros(len(cdf))
            colors = ["#2ca02c", "#ffbf00", "#d62728"]
            for i, col in enumerate(cdf.columns):
                ax.bar(cdf.index, cdf[col], bottom=bottom, label=col.replace("_", " "), color=colors[i])
                bottom = bottom + cdf[col].to_numpy()
            ax.set_title("Domain Coverage by Model")
            ax.set_ylabel("Number of Tasks")
            ax.legend(frameon=False)
            out = dirs["main_figures"] / "07_domain_coverage_stacked_bar.png"
            _savefig(out)
            made.append(str(out))

    # 8) Runtime control regime figure (M4/M5)
    m4_regimes = []
    for name, exp in exps.items():
        if "_M4_" in name and "__regime_" in name:
            s = exp["summary"]
            m4_regimes.append(
                {
                    "regime": exp["experiment_tag"].replace("regime_", ""),
                    "runtime_ms": float(s.get("runtime_ms", np.nan)),
                    "joules": float(s.get("joules", np.nan)),
                    "sla_violation_rate": float(s.get("sla_violation_rate", np.nan)),
                    "policy": "M4",
                }
            )
    m5_runtime = next((n for n in exps if "_M5_" in n and "__runtime_vs_static_comparison" in n), None)
    if m4_regimes:
        rdf = pd.DataFrame(m4_regimes)
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
        sns.barplot(data=rdf, x="regime", y="runtime_ms", ax=axes[0], color="#4C78A8")
        axes[0].set_title("Runtime (ms)")
        sns.barplot(data=rdf, x="regime", y="joules", ax=axes[1], color="#54A24B")
        axes[1].set_title("Joules")
        sns.barplot(data=rdf, x="regime", y="sla_violation_rate", ax=axes[2], color="#E45756")
        axes[2].set_title("SLA Violation Rate")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
        if m5_runtime:
            s = exps[m5_runtime]["summary"]
            axes[0].axhline(float(s.get("runtime_ms", np.nan)), ls="--", color="black", lw=1, label="M5 ref")
            axes[1].axhline(float(s.get("joules", np.nan)), ls="--", color="black", lw=1)
            axes[2].axhline(float(s.get("sla_violation_rate", np.nan)), ls="--", color="black", lw=1)
            axes[0].legend(frameon=False, loc="best")
        fig.suptitle("Runtime Control by Regime (M4) + M5 Reference")
        out = dirs["main_figures"] / "08_runtime_control_regime_figure.png"
        _savefig(out)
        made.append(str(out))

    return made


def plot_appendix_figures(exps: Dict[str, dict], core: Dict[str, str], dirs: Dict[str, Path]) -> List[str]:
    made: List[str] = []

    # Aggregate core metrics
    rows = []
    for m, name in core.items():
        df = exps[name]["metrics"]
        if df is None:
            continue
        tmp = df.copy()
        tmp["model"] = m
        rows.append(tmp)
    all_core = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # 9) Difficulty-stratified success heatmap
    if not all_core.empty:
        if "task_level" in all_core.columns and all_core["task_level"].notna().any():
            labels = ["L1 (easy)", "L2 (medium)", "L3 (hard)", "L4 (very hard)"]
            all_core["difficulty_bucket"] = all_core["task_level"].map({1: labels[0], 2: labels[1], 3: labels[2], 4: labels[3]})
            all_core["difficulty_bucket"] = pd.Categorical(all_core["difficulty_bucket"], categories=labels, ordered=True)
        elif "task_num" in all_core.columns:
            bins = [0, 5, 10, 15, 1000]
            labels = ["1-5", "6-10", "11-15", "16-20+"]
            all_core["difficulty_bucket"] = pd.cut(all_core["task_num"], bins=bins, labels=labels, right=True)
        else:
            all_core["difficulty_bucket"] = "all"
        hm = all_core.groupby(["difficulty_bucket", "model"], as_index=False)["correct"].mean()
        hm_p = hm.pivot(index="difficulty_bucket", columns="model", values="correct").fillna(0.0)
        hm_p = hm_p[[c for c in ["M0", "M1", "M2", "M3", "M4", "M5"] if c in hm_p.columns]]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        sns.heatmap(hm_p, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title("Difficulty-Stratified Success Heatmap")
        out = dirs["appendix_figures"] / "09_difficulty_stratified_success_heatmap.png"
        _savefig(out)
        made.append(str(out))

    # 10) Seed stability fan plot (reward_mean over model progression)
    seq = [m for m in ["M0", "M1", "M2", "M3"] if m in core]
    if seq:
        vals = []
        for i, m in enumerate(seq):
            ps = exps[core[m]]["per_seed"]
            if ps is None or ps.empty or "mean_reward" not in ps.columns:
                continue
            for _, r in ps.iterrows():
                vals.append({"step": i + 1, "model": m, "reward_mean": float(r["mean_reward"]), "seed": int(r["seed"])})
        if vals:
            sdf = pd.DataFrame(vals)
            agg = sdf.groupby("step", as_index=False).agg(mean=("reward_mean", "mean"), lo=("reward_mean", "min"), hi=("reward_mean", "max"))
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(agg["step"], agg["mean"], marker="o", color="#1f77b4")
            ax.fill_between(agg["step"], agg["lo"], agg["hi"], alpha=0.25, color="#1f77b4")
            ax.set_xticks(agg["step"])
            ax.set_xticklabels([seq[i - 1] for i in agg["step"]])
            ax.set_ylabel("Reward Mean")
            ax.set_title("Seed Stability Fan Plot")
            ax.grid(alpha=0.3)
            out = dirs["appendix_figures"] / "10_seed_stability_fan_plot.png"
            _savefig(out)
            made.append(str(out))

    # 11) Efficiency scaling curve (proxy from data-scale experiment prefixes)
    ds_name = next((n for n in exps if "__data_scale_ablation" in n), None)
    if ds_name:
        ddf = exps[ds_name]["metrics"]
        if ddf is not None and not ddf.empty:
            points = []
            if "data_scale_proxy" in ddf.columns and ddf["data_scale_proxy"].notna().any():
                sdf = ddf.copy()
                sdf["data_scale_proxy"] = pd.to_numeric(sdf["data_scale_proxy"], errors="coerce")
                sdf = sdf.dropna(subset=["data_scale_proxy"])
                if not sdf.empty:
                    for n, g in sdf.groupby("data_scale_proxy", as_index=False):
                        points.append(
                            {
                                "num_tasks_proxy": int(n),
                                "correctness": g["correct"].mean(),
                                "joules": g["joules"].mean(),
                                "reward": g["reward"].mean(),
                            }
                        )
            elif "task_num" in ddf.columns:
                for n in [4, 8, 12, 16, 20]:
                    sub = ddf[ddf["task_num"] <= n]
                    if sub.empty:
                        continue
                    points.append(
                        {
                            "num_tasks_proxy": n,
                            "correctness": sub["correct"].mean(),
                            "joules": sub["joules"].mean(),
                            "reward": sub["reward"].mean(),
                        }
                    )
            if points:
                pdf = pd.DataFrame(points).sort_values("num_tasks_proxy")
                fig, ax1 = plt.subplots(figsize=(8, 4.8))
                ax1.plot(pdf["num_tasks_proxy"], pdf["correctness"], marker="o", color="#1f77b4", label="correctness")
                ax1.set_xlabel("Num Tasks (proxy)")
                ax1.set_ylabel("Correctness", color="#1f77b4")
                ax2 = ax1.twinx()
                ax2.plot(pdf["num_tasks_proxy"], pdf["joules"], marker="s", color="#2ca02c", label="joules")
                ax2.set_ylabel("Joules", color="#2ca02c")
                ax1.set_title("Efficiency Scaling Curve")
                out = dirs["appendix_figures"] / "11_efficiency_scaling_curve.png"
                _savefig(out)
                made.append(str(out))

    # 12) Inference budget tradeoff curve
    ib_name = next((n for n in exps if "__inference_budget_ablation" in n), None)
    if ib_name:
        ib = exps[ib_name]["metrics"]
        if ib is not None and not ib.empty and "turns_to_success" in ib.columns:
            budgets = []
            for k in [1, 2, 3, 4, 5]:
                solved = (ib["turns_to_success"] > 0) & (ib["turns_to_success"] <= k)
                budgets.append({"budget_turns": k, "pass_at_k": solved.mean(), "joules": ib[solved]["joules"].mean() if solved.any() else np.nan})
            bdf = pd.DataFrame(budgets)
            fig, ax1 = plt.subplots(figsize=(8, 4.8))
            ax1.plot(bdf["budget_turns"], bdf["pass_at_k"], marker="o", color="#1f77b4")
            ax1.set_xlabel("Max Turns")
            ax1.set_ylabel("Pass@k", color="#1f77b4")
            ax2 = ax1.twinx()
            ax2.plot(bdf["budget_turns"], bdf["joules"], marker="s", color="#d62728")
            ax2.set_ylabel("Joules / Query", color="#d62728")
            ax1.set_title("Inference Budget Tradeoff")
            out = dirs["appendix_figures"] / "12_inference_budget_tradeoff_curve.png"
            _savefig(out)
            made.append(str(out))

    # 13) Latency-Energy joint density
    if not all_core.empty:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        hb = ax.hexbin(all_core["runtime_ms"], all_core["joules"], gridsize=35, cmap="viridis", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_title("Latency-Energy Joint Density")
        ax.set_xlabel("Runtime (ms)")
        ax.set_ylabel("Joules")
        out = dirs["appendix_figures"] / "13_latency_energy_joint_density.png"
        _savefig(out)
        made.append(str(out))

    # 14) OOM/Timeout incidence vs config
    ft_rows = []
    for name, exp in exps.items():
        f = exp["failure"]
        if f is None or f.empty or "reason" not in f.columns or "count" not in f.columns:
            continue
        agg = f.groupby("reason", as_index=False)["count"].sum()
        oom = float(agg.loc[agg["reason"] == "oom", "count"].sum())
        timeout = float(agg.loc[agg["reason"] == "timeout", "count"].sum())
        ft_rows.append({"experiment": exp["experiment_tag"], "oom": oom, "timeout": timeout})
    if ft_rows:
        tdf = pd.DataFrame(ft_rows).sort_values("oom", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(tdf))
        ax.bar(x - 0.2, tdf["oom"], width=0.4, label="oom")
        ax.bar(x + 0.2, tdf["timeout"], width=0.4, label="timeout")
        ax.set_xticks(x)
        ax.set_xticklabels(tdf["experiment"], rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("OOM/Timeout Incidence vs Config")
        ax.legend(frameon=False)
        out = dirs["appendix_figures"] / "14_oom_timeout_incidence_vs_config.png"
        _savefig(out)
        made.append(str(out))

    # 15) Calibration plot (reward vs energy proxy)
    if not all_core.empty:
        cdf = all_core[["reward", "joules"]].dropna().copy()
        if not cdf.empty:
            cdf["pred_bin"] = pd.qcut(cdf["reward"], q=min(10, cdf["reward"].nunique()), duplicates="drop")
            cal = cdf.groupby("pred_bin", as_index=False).agg(pred_reward=("reward", "mean"), realized_energy_proxy=("joules", "mean"))
            fig, ax = plt.subplots(figsize=(7.2, 5))
            ax.scatter(cal["pred_reward"], cal["realized_energy_proxy"], s=60)
            z = np.polyfit(cal["pred_reward"], cal["realized_energy_proxy"], deg=1)
            xs = np.linspace(cal["pred_reward"].min(), cal["pred_reward"].max(), 100)
            ax.plot(xs, z[0] * xs + z[1], "--")
            ax.set_xlabel("Predicted Reward (binned mean)")
            ax.set_ylabel("Realized Energy Proxy (Joules)")
            ax.set_title("Calibration Plot: Predicted Reward vs Realized Outcome")
            out = dirs["appendix_figures"] / "15_calibration_plot.png"
            _savefig(out)
            made.append(str(out))

    # 16) Per-task delta forest plot (M2/M3 - M1)
    if all(k in core for k in ["M1", "M2", "M3"]):
        m1 = exps[core["M1"]]["metrics"]
        m2 = exps[core["M2"]]["metrics"]
        m3 = exps[core["M3"]]["metrics"]
        if m1 is not None and m2 is not None and m3 is not None:
            b = m1.groupby("task_id", as_index=False).agg(j1=("joules", "mean"))
            d2 = m2.groupby("task_id", as_index=False).agg(j2=("joules", "mean"))
            d3 = m3.groupby("task_id", as_index=False).agg(j3=("joules", "mean"))
            d = b.merge(d2, on="task_id").merge(d3, on="task_id")
            d["M2_minus_M1"] = d["j2"] - d["j1"]
            d["M3_minus_M1"] = d["j3"] - d["j1"]
            d = d.sort_values("M3_minus_M1")
            fig, ax = plt.subplots(figsize=(8.6, 7))
            y = np.arange(len(d))
            ax.hlines(y, 0, d["M2_minus_M1"], color="#2ca02c", alpha=0.4)
            ax.plot(d["M2_minus_M1"], y, "o", label="M2 - M1", color="#2ca02c")
            ax.hlines(y, 0, d["M3_minus_M1"], color="#d62728", alpha=0.4)
            ax.plot(d["M3_minus_M1"], y, "s", label="M3 - M1", color="#d62728")
            ax.axvline(0, color="black", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(d["task_id"])
            ax.set_xlabel("delta Joules (negative is better)")
            ax.set_title("Per-Task Delta Forest Plot")
            ax.legend(frameon=False)
            out = dirs["appendix_figures"] / "16_per_task_delta_forest_plot.png"
            _savefig(out)
            made.append(str(out))

    # 17) Routing savings curve (proxy)
    m4_name = core.get("M4")
    m5_name = core.get("M5")
    if m4_name and m5_name:
        s4 = exps[m4_name]["summary"]
        s5 = exps[m5_name]["summary"]
        if s4 and s5:
            x = np.linspace(0.5, 1.0, 11)
            base = float(s4.get("joules", np.nan))
            target = float(s5.get("joules", np.nan))
            gain = max(base - target, 0.0)
            y = gain * (x - 0.5) / 0.5
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.plot(x, y, marker="o")
            ax.set_xlabel("Router Accuracy (proxy)")
            ax.set_ylabel("Energy Savings (J)")
            ax.set_title("Routing Savings Curve (Proxy)")
            ax.grid(alpha=0.3)
            out = dirs["appendix_figures"] / "17_routing_savings_curve.png"
            _savefig(out)
            made.append(str(out))

    # 18) Ablation spider chart
    abs_map = {
        "reward_ablation": next((n for n in exps if "__reward_ablation" in n), None),
        "telemetry_realism_ablation": next((n for n in exps if "__telemetry_realism_ablation" in n), None),
        "data_scale_ablation": next((n for n in exps if "__data_scale_ablation" in n), None),
        "inference_budget_ablation": next((n for n in exps if "__inference_budget_ablation" in n), None),
    }
    rows = []
    for label, name in abs_map.items():
        if not name:
            continue
        s = exps[name]["summary"]
        rows.append(
            {
                "ablation": label,
                "correctness": float(s.get("correctness", np.nan)),
                "speedup": float(s.get("speedup", np.nan)),
                "joules_inv": 1.0 / max(float(s.get("joules", np.nan)), 1e-9),
                "runtime_inv": 1.0 / max(float(s.get("runtime_ms", np.nan)), 1e-9),
                "reward": float(s.get("reward_mean", np.nan)),
            }
        )
    if rows:
        rdf = pd.DataFrame(rows).dropna()
        if not rdf.empty:
            metrics = ["correctness", "speedup", "joules_inv", "runtime_inv", "reward"]
            # min-max normalize each metric
            for m in metrics:
                mn, mx = rdf[m].min(), rdf[m].max()
                rdf[m] = 0.5 if mx == mn else (rdf[m] - mn) / (mx - mn)
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            fig = plt.figure(figsize=(8, 7))
            ax = plt.subplot(111, polar=True)
            for _, r in rdf.iterrows():
                vals = [r[m] for m in metrics]
                vals += vals[:1]
                ax.plot(angles, vals, linewidth=1.8, label=r["ablation"])
                ax.fill(angles, vals, alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_yticklabels([])
            ax.set_title("Ablation Spider Chart (Normalized)")
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=False)
            out = dirs["appendix_figures"] / "18_ablation_spider_chart.png"
            _savefig(out)
            made.append(str(out))

    return made


def build_tables(exps: Dict[str, dict], core: Dict[str, str], dirs: Dict[str, Path]) -> List[str]:
    made: List[str] = []

    # Main model comparison table
    rows = []
    for m in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        name = core.get(m)
        if not name:
            continue
        s = exps[name]["summary"]
        rows.append(
            {
                "model": m,
                "experiment": exps[name]["experiment_tag"],
                "compile_rate": s.get("compile_rate"),
                "correctness": s.get("correctness"),
                "pass_at_k": s.get("pass_at_k"),
                "runtime_ms": s.get("runtime_ms"),
                "joules": s.get("joules"),
                "avg_power_w": s.get("avg_power_w"),
                "speedup": s.get("speedup"),
                "sla_violation_rate": s.get("sla_violation_rate"),
                "reward_mean": s.get("reward_mean"),
                "reward_std": s.get("reward_std"),
            }
        )
    main_df = pd.DataFrame(rows)
    if not main_df.empty:
        csv_path = dirs["tables"] / "table_01_main_model_comparison.csv"
        md_path = dirs["tables"] / "table_01_main_model_comparison.md"
        main_df.to_csv(csv_path, index=False)
        made.append(str(csv_path))
        md = ["| " + " | ".join(main_df.columns) + " |", "|" + "|".join(["---"] * len(main_df.columns)) + "|"]
        for _, r in main_df.iterrows():
            md.append("| " + " | ".join(str(r[c]) for c in main_df.columns) + " |")
        md_path.write_text("\n".join(md) + "\n")
        made.append(str(md_path))

    # Ablation summary table
    abs_rows = []
    for name, exp in exps.items():
        if "_M_ALL_" not in name:
            continue
        if not any(x in name for x in ["ablation", "repeatability", "heldout", "difficulty", "cross_hardware"]):
            continue
        s = exp["summary"]
        abs_rows.append(
            {
                "experiment": exp["experiment_tag"],
                "correctness": s.get("correctness"),
                "pass_at_k": s.get("pass_at_k"),
                "runtime_ms": s.get("runtime_ms"),
                "joules": s.get("joules"),
                "speedup": s.get("speedup"),
                "reward_mean": s.get("reward_mean"),
            }
        )
    abs_df = pd.DataFrame(abs_rows).sort_values("experiment") if abs_rows else pd.DataFrame()
    if not abs_df.empty:
        p = dirs["tables"] / "table_02_ablation_suite.csv"
        abs_df.to_csv(p, index=False)
        made.append(str(p))

    # Runtime regime table
    rr = []
    for name, exp in exps.items():
        if "_M4_" in name and "__regime_" in name:
            s = exp["summary"]
            rr.append(
                {
                    "policy": "M4",
                    "regime": exp["experiment_tag"],
                    "runtime_ms": s.get("runtime_ms"),
                    "joules": s.get("joules"),
                    "avg_power_w": s.get("avg_power_w"),
                    "sla_violation_rate": s.get("sla_violation_rate"),
                    "correctness": s.get("correctness"),
                }
            )
    for name, exp in exps.items():
        if "_M5_" in name and "__runtime_vs_static_comparison" in name:
            s = exp["summary"]
            rr.append(
                {
                    "policy": "M5",
                    "regime": "runtime_vs_static",
                    "runtime_ms": s.get("runtime_ms"),
                    "joules": s.get("joules"),
                    "avg_power_w": s.get("avg_power_w"),
                    "sla_violation_rate": s.get("sla_violation_rate"),
                    "correctness": s.get("correctness"),
                }
            )
    rr_df = pd.DataFrame(rr)
    if not rr_df.empty:
        p = dirs["tables"] / "table_03_runtime_regime.csv"
        rr_df.to_csv(p, index=False)
        made.append(str(p))

    # Pairwise improvements (M2/M3 vs M1)
    if all(k in core for k in ["M1", "M2", "M3"]):
        s1 = exps[core["M1"]]["summary"]
        s2 = exps[core["M2"]]["summary"]
        s3 = exps[core["M3"]]["summary"]

        def rel(a, b):
            return (a - b) / b * 100.0 if b else np.nan

        p_df = pd.DataFrame(
            [
                {
                    "comparison": "M2 vs M1",
                    "delta_correctness_abs": float(s2.get("correctness", np.nan)) - float(s1.get("correctness", np.nan)),
                    "delta_joules_pct": rel(float(s2.get("joules", np.nan)), float(s1.get("joules", np.nan))),
                    "delta_runtime_pct": rel(float(s2.get("runtime_ms", np.nan)), float(s1.get("runtime_ms", np.nan))),
                    "delta_reward_abs": float(s2.get("reward_mean", np.nan)) - float(s1.get("reward_mean", np.nan)),
                },
                {
                    "comparison": "M3 vs M1",
                    "delta_correctness_abs": float(s3.get("correctness", np.nan)) - float(s1.get("correctness", np.nan)),
                    "delta_joules_pct": rel(float(s3.get("joules", np.nan)), float(s1.get("joules", np.nan))),
                    "delta_runtime_pct": rel(float(s3.get("runtime_ms", np.nan)), float(s1.get("runtime_ms", np.nan))),
                    "delta_reward_abs": float(s3.get("reward_mean", np.nan)) - float(s1.get("reward_mean", np.nan)),
                },
            ]
        )
        p = dirs["tables"] / "table_04_pairwise_improvements.csv"
        p_df.to_csv(p, index=False)
        made.append(str(p))

    return made


def write_manifest(dirs: Dict[str, Path], main_figs: List[str], app_figs: List[str], tables: List[str]) -> Path:
    manifest = {
        "main_figures": main_figs,
        "appendix_figures": app_figs,
        "tables": tables,
    }
    out = dirs["root"] / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/h100/2026-03"),
        help="Input experiment root directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/h100/2026-03/paper_outputs"),
        help="Output directory for generated figures/tables",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    exps = load_experiments(args.input_root)
    if not exps:
        raise FileNotFoundError(f"No experiments found under: {args.input_root}")

    core = get_core_experiment_names(exps)
    dirs = ensure_dirs(args.output_root)

    main_figs = plot_main_figures(exps, core, dirs)
    app_figs = plot_appendix_figures(exps, core, dirs)
    tables = build_tables(exps, core, dirs)
    manifest = write_manifest(dirs, main_figs, app_figs, tables)

    print(f"Wrote artifacts to {args.output_root}")
    print(f"Main figures: {len(main_figs)}")
    print(f"Appendix figures: {len(app_figs)}")
    print(f"Tables: {len(tables)}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
