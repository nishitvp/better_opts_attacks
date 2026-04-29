# %% [markdown]
# # Phase-transition analysis
#
# Loads all runs from a phase_transition_runs.py experiment folder and produces:
#   1. Survival curve — CDF of crash time across runs
#   2. Aligned logprob trajectories — crash vs plateau groups
#   3. Trajectory metric trajectories — aligned to crash time T*, checking
#      whether payload attention / first-token logprob leads the logprob crash
#
# Run from repo root:
#   python analysis/phase_transition_analysis.py --expt-folder logs/phase_transition_...
# or open in Jupyter as a notebook (cells delimited by # %%).

# %%
import argparse
import json
import os
import shelve
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils.experiment_logger as experiment_logger

# %%  ── CONFIG ────────────────────────────────────────────────────────────────
# When run as a script, read from CLI. When used interactively, set EXPT_FOLDER.
if __name__ == "__main__" and len(sys.argv) > 1:
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--expt-folder", type=str, required=True)
    _parser.add_argument("--logprob-crash-threshold", type=float, default=3.0,
                         help="Logprob loss below this is considered a 'crash' (default 3.0)")
    _parser.add_argument("--target-string", type=str, default="Hacked",
                         help="Target string used to identify success from summary.json")
    _args = _parser.parse_args()
    EXPT_FOLDER = _args.expt_folder
    LOGPROB_CRASH_THRESHOLD = _args.logprob_crash_threshold
else:
    EXPT_FOLDER = "logs/phase_transition_example"   # ← set interactively
    LOGPROB_CRASH_THRESHOLD = 3.0

PLOT_DIR = os.path.join(EXPT_FOLDER, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# %%  ── HELPERS ───────────────────────────────────────────────────────────────

def _load_run(run_dir: str):
    """
    Reconstruct per-step logprobs and trajectory metrics for one run.

    Returns a dict:
        logprobs          : list[float]        — one value per step (step -1 prepended)
        payload_attn_uniform  : list[float|None]
        payload_attn_weighted : list[float|None]
        first_token_logprob   : list[float|None]
        success           : bool|None          — from final generation
        phase_boundaries  : list[int]          — step indices where phase changes
        n_phase1_steps    : int
    """
    meta_path = os.path.join(run_dir, "metadata.jsonl")
    obj_path  = os.path.join(run_dir, "objects")
    if not os.path.exists(meta_path):
        return None

    df = experiment_logger.load_experiment_logs(meta_path, include_trace_stack=True)
    shelf = shelve.open(obj_path)

    try:
        # Get trace IDs for every custom_gcg call (one per phase)
        params_rows = df[
            (df["variable_name"] == "params") &
            (df["function_name"] == "custom_gcg")
        ].sort_values("timestamp")

        if params_rows.empty:
            return None

        phase_trace_ids = params_rows["trace_id"].tolist()

        def _chunks_for_trace(varname, trace_id):
            """Return list of (step_num, chunk_object) sorted by step_num."""
            rows = df[
                (df["variable_name"] == varname) &
                (df.get("trace_stack.1", pd.Series(dtype=object)) == trace_id)
            ].sort_values("step_num")
            results = []
            for _, row in rows.iterrows():
                obj = shelf.get(row["id"])
                if obj is not None:
                    results.append((row["step_num"], obj))
            return results

        logprobs_all, attn_u_all, attn_w_all, ftlp_all = [], [], [], []
        n_phase1_steps = 0

        for phase_idx, trace_id in enumerate(phase_trace_ids):
            # step -1 (initial) — only for phase 0
            if phase_idx == 0:
                for _, row in df[
                    (df["variable_name"] == "initial_logprobs") &
                    (df.get("trace_stack.1", pd.Series(dtype=object)) == trace_id)
                ].iterrows():
                    v = shelf.get(row["id"])
                    if v is not None:
                        logprobs_all.append(v)
                for _, row in df[
                    (df["variable_name"] == "initial_trajectory_metrics") &
                    (df.get("trace_stack.1", pd.Series(dtype=object)) == trace_id)
                ].iterrows():
                    v = shelf.get(row["id"])
                    if v is not None and isinstance(v, dict):
                        attn_u_all.append(v.get("payload_attn_uniform"))
                        attn_w_all.append(v.get("payload_attn_weighted"))
                        ftlp_all.append(v.get("first_token_logprob"))

            lp_chunks = _chunks_for_trace("logprobs_chunk", trace_id)
            tm_chunks = _chunks_for_trace("trajectory_metrics_chunk", trace_id)

            for _, chunk in lp_chunks:
                logprobs_all.extend(chunk)
            for _, chunk in tm_chunks:
                for m in chunk:
                    attn_u_all.append(m.get("payload_attn_uniform") if m else None)
                    attn_w_all.append(m.get("payload_attn_weighted") if m else None)
                    ftlp_all.append(m.get("first_token_logprob") if m else None)

            if phase_idx == 0:
                n_phase1_steps = len(logprobs_all)

        # Success label from logged bool
        success = None
        for _, row in df[df["variable_name"] == "success"].iterrows():
            v = shelf.get(row["id"])
            if isinstance(v, bool):
                success = v
                break

    finally:
        shelf.close()

    return {
        "logprobs": logprobs_all,
        "payload_attn_uniform": attn_u_all,
        "payload_attn_weighted": attn_w_all,
        "first_token_logprob": ftlp_all,
        "success": success,
        "n_phase1_steps": n_phase1_steps,
    }


def load_all_runs(expt_folder: str):
    run_dirs = sorted([
        os.path.join(expt_folder, d)
        for d in os.listdir(expt_folder)
        if d.startswith("run_") and os.path.isdir(os.path.join(expt_folder, d))
    ])
    runs = []
    for run_dir in run_dirs:
        r = _load_run(run_dir)
        if r is not None:
            r["run_dir"] = run_dir
            runs.append(r)
    print(f"Loaded {len(runs)} runs from {expt_folder}")
    return runs


def find_crash_time(logprobs, threshold=LOGPROB_CRASH_THRESHOLD):
    """Return first step index where logprob drops below threshold, else None."""
    for i, lp in enumerate(logprobs):
        if lp is not None and lp < threshold:
            return i
    return None


# %%  ── LOAD ───────────────────────────────────────────────────────────────────
runs = load_all_runs(EXPT_FOLDER)

for r in runs:
    r["crash_time"] = find_crash_time(r["logprobs"], LOGPROB_CRASH_THRESHOLD)
    r["crashed"] = r["crash_time"] is not None

crash_runs   = [r for r in runs if r["crashed"]]
plateau_runs = [r for r in runs if not r["crashed"]]

n_total   = len(runs)
n_crashed = len(crash_runs)
print(f"\nCrash/success breakdown:")
print(f"  Crashed (logprob < {LOGPROB_CRASH_THRESHOLD}): {n_crashed}/{n_total}")
print(f"  Success (model output):  {sum(r['success'] is True for r in runs)}/{n_total}")
if crash_runs:
    crash_times = [r["crash_time"] for r in crash_runs]
    print(f"  Crash times — median={np.median(crash_times):.0f}, "
          f"mean={np.mean(crash_times):.0f}, "
          f"min={min(crash_times)}, max={max(crash_times)}")

# %%  ── PLOT 1: Survival curve ─────────────────────────────────────────────────
if crash_runs:
    crash_times_sorted = sorted(r["crash_time"] for r in crash_runs)
    phase1_end = int(np.median([r["n_phase1_steps"] for r in runs]))

    fig, ax = plt.subplots(figsize=(7, 4))
    cdf_x = [0] + crash_times_sorted + [max(len(r["logprobs"]) for r in runs)]
    cdf_y = [0] + [i / n_total for i in range(1, n_crashed + 1)] + [n_crashed / n_total]
    ax.step(cdf_x, cdf_y, where="post", color="steelblue", lw=2)
    ax.axvline(phase1_end, color="orange", ls="--", lw=1.5, label=f"Phase 1 end (~step {phase1_end})")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Fraction of runs crashed")
    ax.set_title(f"Survival curve  ({n_crashed}/{n_total} crash by end)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "survival_curve.png"), dpi=150)
    plt.show()
    print(f"Saved survival_curve.png")
else:
    print("No crash runs — skipping survival curve.")

# %%  ── PLOT 2: Mean logprob trajectories, crash vs plateau ───────────────────
MAX_STEPS = max(len(r["logprobs"]) for r in runs)

def _pad(seq, length, fill=np.nan):
    arr = np.full(length, fill)
    arr[:len(seq)] = [x if x is not None else np.nan for x in seq]
    return arr

fig, ax = plt.subplots(figsize=(9, 4))
steps = np.arange(MAX_STEPS)

if crash_runs:
    lp_crash = np.array([_pad(r["logprobs"], MAX_STEPS) for r in crash_runs])
    mean_c = np.nanmean(lp_crash, axis=0)
    std_c  = np.nanstd(lp_crash, axis=0)
    ax.plot(steps, mean_c, color="steelblue", lw=2, label=f"crash  (n={n_crashed})")
    ax.fill_between(steps, mean_c - std_c, mean_c + std_c, alpha=0.2, color="steelblue")

if plateau_runs:
    lp_plat = np.array([_pad(r["logprobs"], MAX_STEPS) for r in plateau_runs])
    mean_p = np.nanmean(lp_plat, axis=0)
    std_p  = np.nanstd(lp_plat, axis=0)
    ax.plot(steps, mean_p, color="tomato", lw=2, label=f"plateau (n={len(plateau_runs)})")
    ax.fill_between(steps, mean_p - std_p, mean_p + std_p, alpha=0.2, color="tomato")

if runs:
    phase1_end = int(np.median([r["n_phase1_steps"] for r in runs]))
    ax.axvline(phase1_end, color="orange", ls="--", lw=1.5, label=f"Phase 1 end")

ax.axhline(LOGPROB_CRASH_THRESHOLD, color="gray", ls=":", lw=1, label=f"crash threshold={LOGPROB_CRASH_THRESHOLD}")
ax.set_xlabel("Step")
ax.set_ylabel("Logprob loss (lower = better)")
ax.set_title("Logprob trajectories: crash vs plateau groups")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "logprob_trajectories.png"), dpi=150)
plt.show()

# %%  ── PLOT 3: Payload attention — crash vs plateau, standard alignment ───────
has_attn = any(
    r["payload_attn_uniform"] and any(v is not None for v in r["payload_attn_uniform"])
    for r in runs
)
if has_attn:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in [
        (axes[0], "payload_attn_uniform",  "Payload attention (uniform)"),
        (axes[1], "first_token_logprob",   "First-token log-prob"),
    ]:
        for group, color, name in [
            (crash_runs,   "steelblue", f"crash (n={n_crashed})"),
            (plateau_runs, "tomato",    f"plateau (n={len(plateau_runs)})"),
        ]:
            if not group:
                continue
            arr = np.array([_pad(r[key], MAX_STEPS) for r in group])
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr, axis=0)
            ax.plot(steps, mean, color=color, lw=2, label=name)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
        if runs:
            ax.axvline(phase1_end, color="orange", ls="--", lw=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "attention_trajectories.png"), dpi=150)
    plt.show()

# %%  ── PLOT 4: T*-aligned trajectories (precursor test) ──────────────────────
# For crash runs: align each trajectory to crash time T*, showing steps [-W, +W].
# Key question: does payload_attn / first_token_logprob diverge from plateau runs
# BEFORE T*, meaning it precedes rather than coincides with the crash?
WINDOW = 30  # steps before/after crash

if len(crash_runs) >= 5 and has_attn:
    for metric_key, metric_label in [
        ("payload_attn_uniform", "Payload attention (uniform)"),
        ("first_token_logprob",  "First-token log-prob"),
        ("logprobs",             "Logprob loss"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        offset_range = np.arange(-WINDOW, WINDOW + 1)

        # Crash runs: extract window around T*
        aligned = []
        for r in crash_runs:
            T = r["crash_time"]
            seq = r[metric_key]
            window_vals = []
            for offset in offset_range:
                idx = T + offset
                if 0 <= idx < len(seq) and seq[idx] is not None:
                    window_vals.append(seq[idx])
                else:
                    window_vals.append(np.nan)
            aligned.append(window_vals)
        aligned = np.array(aligned)
        mean_a = np.nanmean(aligned, axis=0)
        std_a  = np.nanstd(aligned, axis=0)
        ax.plot(offset_range, mean_a, color="steelblue", lw=2,
                label=f"crash runs aligned to T* (n={len(crash_runs)})")
        ax.fill_between(offset_range, mean_a - std_a, mean_a + std_a,
                        alpha=0.2, color="steelblue")

        # Plateau runs: use their best step as pseudo-T* (lowest logprob step)
        if plateau_runs:
            pseudo_aligned = []
            for r in plateau_runs:
                lp = r["logprobs"]
                T_pseudo = int(np.nanargmin([x if x is not None else np.inf for x in lp]))
                seq = r[metric_key]
                window_vals = []
                for offset in offset_range:
                    idx = T_pseudo + offset
                    if 0 <= idx < len(seq) and seq[idx] is not None:
                        window_vals.append(seq[idx])
                    else:
                        window_vals.append(np.nan)
                pseudo_aligned.append(window_vals)
            pseudo_aligned = np.array(pseudo_aligned)
            mean_pa = np.nanmean(pseudo_aligned, axis=0)
            std_pa  = np.nanstd(pseudo_aligned, axis=0)
            ax.plot(offset_range, mean_pa, color="tomato", lw=2, ls="--",
                    label=f"plateau runs (pseudo-T*, n={len(plateau_runs)})")
            ax.fill_between(offset_range, mean_pa - std_pa, mean_pa + std_pa,
                            alpha=0.15, color="tomato")

        ax.axvline(0, color="gray", ls=":", lw=1.5, label="T* (crash)")
        ax.set_xlabel("Step offset from T*")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} — aligned to crash time T*")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = metric_key.replace("/", "_")
        fig.savefig(os.path.join(PLOT_DIR, f"aligned_{fname}.png"), dpi=150)
        plt.show()
else:
    print("Fewer than 5 crash runs or no attention data — skipping T*-aligned plots.")

# %%  ── QUANTITATIVE: lead/lag between attention and crash ────────────────────
# For each crash run: find when payload_attn_uniform first exceeds its run-median.
# Report mean offset between attention threshold crossing and T*.
if len(crash_runs) >= 5 and has_attn:
    leads = []
    for r in crash_runs:
        attn = r["payload_attn_uniform"]
        if not attn or all(v is None for v in attn):
            continue
        attn_clean = [v if v is not None else np.nan for v in attn]
        threshold = np.nanmedian(attn_clean)
        T = r["crash_time"]
        # Find first step where attn exceeds threshold
        cross = next(
            (i for i, v in enumerate(attn_clean) if not np.isnan(v) and v > threshold),
            None,
        )
        if cross is not None:
            leads.append(T - cross)  # positive = attn crossed before crash

    if leads:
        print(f"\nAttention threshold-crossing lead over logprob crash:")
        print(f"  Mean lead  = {np.mean(leads):.1f} steps  "
              f"(positive = attention led the crash)")
        print(f"  Median     = {np.median(leads):.1f}")
        print(f"  Std        = {np.std(leads):.1f}")
        print(f"  Fraction with positive lead: "
              f"{sum(l > 0 for l in leads)}/{len(leads)}")

# %%
print("\nAnalysis complete. Plots saved to", PLOT_DIR)
