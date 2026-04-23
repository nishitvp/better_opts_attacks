#!/usr/bin/env python3
"""
run_profiling.py
================

Load successful injections from CSV/JSON and run perturbation profiling.

Input sources (pick one):
  --input-csv   matched_with_combos.csv  (suite, user_task_id, injection_task_id,
                                          injection, best_score columns)
  --input-json  legacy JSON with {injection, best_match} entries

Usage:
    python run_profiling.py --input-csv ../data/agentdojo_data/matched_with_combos.csv \
        --min-score 0.5 --model openai/gpt-oss-20b

    python run_profiling.py --input-csv ../data/agentdojo_data/matched_with_combos.csv \
        --min-score 0.5 --device cuda:1 --dry-run
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from agentdojo.task_suite import get_suite

sys.path.append("..")

from utils.experiment_logger_sql import ExperimentLogger
from profiling_logger import ProfilingLogger
from inference_runner import LocalHarmonyLLM, build_pipeline
from feature_hooks import FeatureConfig
from perturbation_profiling import run_first_token_experiment, DISTANCE_DEFAULTS

logger = logging.getLogger(__name__)


def load_inputs_from_csv(path: str, min_score: float = 0.0,
                         version: str = "v1.2",
                         dedup_payloads: bool = False,
                         seed: int = 42) -> list:
    """Load from matched_with_combos.csv.

    Columns: injection, best_match_label, best_match_goal, best_score,
             suite, user_task_id, user_task_prompt, injection_task_id,
             injection_task_goal

    Filters by best_score >= min_score and validates that the suite/tasks
    exist in AgentDojo.

    If dedup_payloads=True, keeps one randomly sampled user_task per
    (payload, suite, injection_task_id). Seed controls the sampling.
    """
    import random
    rng = random.Random(seed)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Build all valid candidates first
    from collections import defaultdict
    candidates = defaultdict(list)   # key -> list of input dicts
    skipped = 0

    for row in rows:
        score = float(row.get("best_score", 0))
        if score < min_score:
            skipped += 1
            continue

        suite_name = row.get("suite", "").strip()
        user_task_id = row.get("user_task_id", "").strip()
        injection_task_id = row.get("injection_task_id", "").strip()
        payload = row.get("injection", "").strip()

        if not all([suite_name, user_task_id, injection_task_id, payload]):
            skipped += 1
            continue

        try:
            suite = get_suite(version, suite_name)
        except Exception:
            skipped += 1
            continue

        if user_task_id not in suite.user_tasks:
            skipped += 1
            continue
        if injection_task_id not in suite.injection_tasks:
            skipped += 1
            continue

        entry = {
            "suite": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "payload": payload,
            "best_score": score,
            "best_match_goal": row.get("best_match_goal", ""),
        }
        key = (payload, suite_name, injection_task_id)
        candidates[key].append(entry)

    if dedup_payloads:
        inputs = [rng.choice(group) for group in candidates.values()]
    else:
        inputs = [entry for group in candidates.values() for entry in group]

    logger.info("Loaded %d inputs from CSV (skipped %d, min_score=%.2f, dedup=%s)",
                len(inputs), skipped, min_score, dedup_payloads)
    return inputs


def load_inputs_varied_contexts(path: str, version: str = "v1.2") -> list:
    """Load fixed payloads from CSV and expand to all user_tasks in each suite.

    Reads profiling_selected.csv (one row per canonical injection) as the
    source of (payload, suite, injection_task_id) triples, then crosses each
    with every user_task the suite supports.  Duplicate (suite, injection_task_id)
    rows are collapsed to the first payload seen.

    The result is a flat list of input dicts ready for run_first_token_experiment.
    Metadata logged per run (suite, user_task_id, injection_task_id) is sufficient
    to group results by injection vs. context in analysis.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    seen: dict = {}          # (suite, injection_task_id) → input row
    skipped = 0
    for row in rows:
        suite_name       = row.get("suite", "").strip()
        injection_task_id = row.get("injection_task_id", "").strip()
        payload          = row.get("injection", "").strip()

        if not all([suite_name, injection_task_id, payload]):
            skipped += 1
            continue

        key = (suite_name, injection_task_id)
        if key in seen:
            logger.warning("Duplicate (suite, injection_task_id) %s — keeping first payload", key)
            continue
        seen[key] = row

    inputs = []
    for (suite_name, injection_task_id), row in seen.items():
        try:
            suite = get_suite(version, suite_name)
        except Exception:
            logger.warning("Suite %s not found in AgentDojo %s — skipping", suite_name, version)
            skipped += 1
            continue

        if injection_task_id not in suite.injection_tasks:
            logger.warning("%s/%s not in AgentDojo — skipping", suite_name, injection_task_id)
            skipped += 1
            continue

        payload = row["injection"].strip()
        for user_task_id in sorted(suite.user_tasks):
            inputs.append({
                "suite":             suite_name,
                "user_task_id":      user_task_id,
                "injection_task_id": injection_task_id,
                "payload":           payload,
                "best_score":        float(row.get("best_score", 0)),
                "best_match_goal":   row.get("best_match_goal", ""),
            })

    logger.info(
        "Varied-context mode: %d injections × all user_tasks → %d inputs "
        "(%d skipped)",
        len(seen), len(inputs), skipped,
    )
    return inputs


def load_inputs_from_json(path: str, version: str = "v1.2") -> list:
    """Load from legacy JSON with {injection, best_match} format."""
    with open(path) as f:
        raw = json.load(f)

    inputs = []
    skipped = 0
    for entry in raw:
        payload = entry["injection"]
        best_match = entry["best_match"]

        parts = best_match.split("/")
        if len(parts) != 3 or parts[1] != "injection":
            skipped += 1
            continue

        suite_name = parts[0]
        injection_task_id = parts[2]

        try:
            suite = get_suite(version, suite_name)
        except Exception:
            skipped += 1
            continue

        if injection_task_id not in suite.injection_tasks:
            skipped += 1
            continue

        for user_task_id in suite.user_tasks:
            inputs.append({
                "suite": suite_name,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "payload": payload,
                "best_score": entry.get("best_score", 0.0),
                "best_match_goal": entry.get("best_match_goal", ""),
            })

    logger.info("Loaded %d (payload, user_task) pairs from JSON (%d skipped)",
                len(inputs), skipped)
    return inputs


def main():
    p = argparse.ArgumentParser(description="Run first-token perturbation profiling")

    # Input
    inp_grp = p.add_mutually_exclusive_group(required=True)
    inp_grp.add_argument("--input-csv", help="matched_with_combos.csv")
    inp_grp.add_argument("--input-json", help="Legacy injections JSON")
    p.add_argument("--min-score", type=float, default=0.4,
                   help="Minimum best_score to include (CSV only, default 0.4)")
    p.add_argument("--dedup-payloads", action="store_true",
                   help="Keep only one user_task per (payload, suite, injection_task_id). "
                        "Collapses the CSV cross-join to one experiment per distinct payload.")
    p.add_argument("--vary-contexts", action="store_true",
                   help="Fixed-payload, varied-context mode. "
                        "Reads one canonical (payload, injection_task_id) per suite from "
                        "--input-csv, then expands to every user_task in the suite. "
                        "Incompatible with --dedup-payloads.")
    p.add_argument("--n-contexts", type=int, default=None,
                   help="Randomly sample this many inputs from the expanded list, "
                        "stratified by injection so each is represented equally. "
                        "Only meaningful with --vary-contexts.")

    # Model
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--version", default="v1.2")

    # Output
    p.add_argument("--log-dir", default=None,
                   help="Output directory (default: profiling_logs/YYYYMMDD_HHMMSS)")

    # Grid
    p.add_argument("--n-values", type=int, nargs="*", default=None)
    p.add_argument("--target-per-cell", type=int, default=32)
    p.add_argument("--max-candidates", type=int, default=256)
    p.add_argument("--distance-fn", default="semantic",
                   choices=["semantic", "fluency"])
    p.add_argument("--distance-min", type=float, default=None,
                   help="Lower bound for distance filter (default: metric-specific)")
    p.add_argument("--distance-max", type=float, default=None,
                   help="Upper bound for distance filter (default: metric-specific)")
    p.add_argument("--distance-sweep", action="store_true",
                   help="Run a sweep of distance bands instead of a single range. "
                        "Bands are defined per metric (fluency: 0-1, 1-2, 2-3, 3-5, 5-8). "
                        "Each band gets its own subfolder. Incompatible with --distance-min/max.")
    p.add_argument("--perturbation-types", nargs="+",
                   choices=["delete", "flip"], default=None,
                   help="Perturbation types to run (default: delete flip)")

    # Profiling layers
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--n-reasoning-tokens", type=int, default=10,
                   help="Number of first reasoning tokens to capture attention at")

    # Extra signal capture (off by default for backwards compatibility)
    p.add_argument("--capture-value-aggregates", action="store_true",
                   help="Capture per-head value aggregates sum_i(alpha_i v_i) before o_proj")
    p.add_argument("--capture-mlp-output", action="store_true",
                   help="Capture MLP/MoE additive contribution to residual stream per layer")
    p.add_argument("--capture-logit-lens", action="store_true",
                   help="Project hidden states through lm_head at each layer (logit lens)")
    p.add_argument("--logit-lens-top-k", type=int, default=50,
                   help="Top-k tokens to store per position per layer for logit lens (default: 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-successes-per-cell", type=int, default=None,
                   help="Cap on logged successes per perturbation cell. Once reached, "
                        "further successes are skipped (pipeline still runs, attention "
                        "capture and DB write are skipped). Failures are always logged. "
                        "Default: no cap (log everything).")
    p.add_argument("--failure-tree-n-roots", type=int, default=0,
                   help="Branch from this many diverse level-1 failures to generate "
                        "additional failure examples (addresses class imbalance). "
                        "0 = disabled (default).")
    p.add_argument("--failure-tree-levels", type=int, default=1,
                   help="Number of tree levels to grow from the initial failure seeds. "
                        "1 = single level (default, current behaviour). "
                        "Each subsequent level seeds from the previous level's failures.")
    p.add_argument("--failure-tree-budget-decay", type=float, default=0.5,
                   help="Multiply target_per_cell by this factor at each successive tree "
                        "level (default 0.5). Keeps total run count bounded as depth grows.")

    p.add_argument("--baseline-only", action="store_true",
                   help="Run only the baseline pipeline for each input and report "
                        "success/failure. No perturbation grid. Use this to screen "
                        "inputs before committing to a full experiment.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--vital-strings-json", default=None,
                   help="Path to JSON file mapping injection_task_id → list of strings "
                        "that must not be perturbed (e.g. account numbers, email addresses). "
                        "Example: {\"injection_task_7\": [\"1234567890\", \"victim@example.com\"]}")
    args = p.parse_args()

    if args.distance_sweep and (args.distance_min is not None or args.distance_max is not None):
        p.error("--distance-sweep cannot be combined with --distance-min / --distance-max")
    if args.vary_contexts and args.dedup_payloads:
        p.error("--vary-contexts and --dedup-payloads are mutually exclusive")
    if args.vary_contexts and not args.input_csv:
        p.error("--vary-contexts requires --input-csv")

    perturbation_types = args.perturbation_types or ["delete", "flip"]

    # Per-metric sweep bands (fluency-centric; semantic has a narrower natural range)
    SWEEP_BANDS = {
        "fluency":  [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0)],
        "semantic": [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8)],
    }

    # Build the list of (d_min, d_max) to run
    if args.distance_sweep:
        distance_ranges = SWEEP_BANDS[args.distance_fn]
    else:
        d_min, d_max = DISTANCE_DEFAULTS[args.distance_fn]
        if args.distance_min is not None:
            d_min = args.distance_min
        if args.distance_max is not None:
            d_max = args.distance_max
        distance_ranges = [(d_min, d_max)]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-25s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["transformers", "torch", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── Load inputs ────────────────────────────────────────────────
    if args.input_csv:
        if args.vary_contexts:
            inputs = load_inputs_varied_contexts(args.input_csv, args.version)
        else:
            inputs = load_inputs_from_csv(args.input_csv, args.min_score, args.version,
                                          dedup_payloads=args.dedup_payloads,
                                          seed=args.seed)
    else:
        inputs = load_inputs_from_json(args.input_json, args.version)

    # ── Attach vital strings ───────────────────────────────────────
    # vital_strings_map keys: "suite/injection_task_id" (preferred) or
    # bare "injection_task_id" (fallback for suites without conflicts).
    # These strings are never perturbed to avoid trivial failures where the
    # target of the injection (e.g. account number, email) changes.
    if args.vital_strings_json:
        with open(args.vital_strings_json) as _f:
            vital_map = json.load(_f)
        for inp in inputs:
            composite = f"{inp['suite']}/{inp['injection_task_id']}"
            vs = vital_map.get(composite) or vital_map.get(inp["injection_task_id"], [])
            if vs:
                inp["vital_strings"] = vs
        logger.info("Loaded vital strings for %d injection task IDs", len(vital_map))

    if not inputs:
        logger.error("No valid inputs loaded")
        sys.exit(1)

    # ── Optional stratified subsample ──────────────────────────────
    if args.n_contexts is not None:
        import random
        from collections import defaultdict
        rng = random.Random(args.seed)
        # Group by (suite, injection_task_id) so every injection is represented
        groups: dict = defaultdict(list)
        for inp in inputs:
            groups[(inp["suite"], inp["injection_task_id"])].append(inp)
        n_groups = len(groups)
        base, remainder = divmod(args.n_contexts, n_groups)
        sampled = []
        for gi, (key, group) in enumerate(sorted(groups.items())):
            k = base + (1 if gi < remainder else 0)
            k = min(k, len(group))
            sampled.extend(rng.sample(group, k))
        total_before = sum(len(g) for g in groups.values())
        inputs = sampled
        logger.info(
            "Sampled %d/%d contexts (%d injection groups, seed=%d)",
            len(inputs), total_before, n_groups, args.seed,
        )

    # ── Resolve root log dir ───────────────────────────────────────
    # Auto-generate a timestamped subfolder unless the user pinned a path.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_log_dir = Path(args.log_dir) if args.log_dir else Path("profiling_logs") / run_stamp

    # ── Dry run ────────────────────────────────────────────────────
    if args.dry_run:
        from collections import Counter
        suites = set(i["suite"] for i in inputs)
        n_values = args.n_values or [20, 40, 60, 80, 100, 120, 140, 160]
        cells_per_band = len(n_values) * 3 * len(perturbation_types)
        total_cells = cells_per_band * len(inputs) * len(distance_ranges)
        max_runs = total_cells * args.target_per_cell + len(inputs) * len(distance_ranges)
        inj_counts = Counter(f"{i['suite']}/{i['injection_task_id']}" for i in inputs)
        print(f"Mode:        {'varied-context' if args.vary_contexts else 'standard'}")
        print(f"Inputs:      {len(inputs)} (payload, user_task) pairs")
        print(f"Suites:      {sorted(suites)}")
        for inj, cnt in sorted(inj_counts.items()):
            print(f"  {inj}: {cnt} user_task contexts")
        print(f"Types:       {perturbation_types}")
        print(f"Distance fn: {args.distance_fn}")
        print(f"Bands:       {distance_ranges}")
        print(f"Target/cell: {args.target_per_cell}")
        print(f"N values:    {n_values}")
        print(f"Grid:        {cells_per_band} cells/band, {total_cells} total cells")
        print(f"Max runs:    {max_runs} (upper bound, gated by baseline success)")
        print(f"Log dir:     {root_log_dir}/")
        return

    # ── Build LLM (once, shared across all bands) ──────────────────
    llm = LocalHarmonyLLM(
        model_name=args.model,
        device=args.device,
        reasoning_effort="low",
    )

    n_layers = len(llm.feature_capture._layer_modules)
    if args.layers:
        profile_layers = sorted(set(args.layers))
    else:
        profile_layers = sorted(range(0, n_layers))

    llm.feature_config = FeatureConfig(
        layers=profile_layers,
        capture_attention=True,
        capture_hidden_states=True,
        capture_value_aggregates=args.capture_value_aggregates,
        capture_mlp_output=args.capture_mlp_output,
        capture_logit_lens=args.capture_logit_lens,
        logit_lens_top_k=args.logit_lens_top_k,
        attention_last_row_only=True,
    )

    logger.info("Model: %s  (%d layers, profiling %s)",
                args.model, n_layers, profile_layers)

    pipeline = build_pipeline(llm)

    # ── Baseline-only screening ────────────────────────────────────
    if args.baseline_only:
        from perturbation_profiling import profile_first_token_only
        print(f"\nBaseline screening: {len(inputs)} inputs\n")
        succeeded, failed = [], []
        for i, inp in enumerate(inputs):
            label = f"{inp['suite']}/{inp['user_task_id']}/{inp['injection_task_id']}"
            out = profile_first_token_only(
                llm, pipeline,
                inp["suite"], inp["user_task_id"], inp["injection_task_id"],
                inp["payload"], llm.feature_config,
                n_reasoning_tokens=args.n_reasoning_tokens,
                run_label=f"baseline_{i}",
                version=args.version,
            )
            if out["task_result"] is None:
                result = "ERROR"
                failed.append(inp)
            else:
                ev = out["task_result"].evaluation
                ok = bool(ev.get("injection_success"))
                result = "OK " if ok else "FAIL"
                (succeeded if ok else failed).append(inp)
            print(f"  [{i+1:3d}/{len(inputs)}] {result}  {label}  "
                  f"score={inp['best_score']:.2f}  ({out['wall_time']:.1f}s)")

        print(f"\nSucceeded: {len(succeeded)}/{len(inputs)}")
        print(f"Failed:    {len(failed)}/{len(inputs)}")
        if succeeded:
            print("\nSuccessful inputs:")
            for inp in succeeded:
                print(f"  {inp['suite']}/{inp['user_task_id']}/{inp['injection_task_id']}  "
                      f"score={inp['best_score']:.2f}")
        return

    # ── Run one band (or all bands in a sweep) ─────────────────────
    all_summaries = []
    for band_idx, (d_min, d_max) in enumerate(distance_ranges):
        band_label = f"d{d_min:.2g}-{d_max:.2g}"

        if len(distance_ranges) > 1:
            log_dir = root_log_dir / band_label
            logger.info("=" * 60)
            logger.info("Band %d/%d: %s in [%.3g, %.3g]",
                        band_idx + 1, len(distance_ranges),
                        args.distance_fn, d_min, d_max)
        else:
            log_dir = root_log_dir

        log_dir.mkdir(parents=True, exist_ok=True)

        exp_logger = ExperimentLogger(str(log_dir))
        prof_logger = ProfilingLogger(exp_logger, tensor_dir=str(log_dir / "tensors"))

        config = {
            "perturbation_types": perturbation_types,
            "n_values": args.n_values or [20, 40, 60, 80, 100, 120, 140, 160],
            "target_per_cell": args.target_per_cell,
            "max_candidates_per_cell": args.max_candidates,
            "distance_fn_name": args.distance_fn,
            "distance_range": [d_min, d_max],
            "n_reasoning_tokens": args.n_reasoning_tokens,
            "seed": args.seed,
            "agentdojo_version": args.version,
            "max_successes_per_cell": args.max_successes_per_cell,
            "failure_tree_n_roots": args.failure_tree_n_roots,
            "failure_tree_levels": args.failure_tree_levels,
            "failure_tree_budget_decay": args.failure_tree_budget_decay,
        }

        summary = run_first_token_experiment(llm, pipeline, inputs, config, prof_logger)
        summary["band_label"] = band_label
        summary["distance_range"] = [d_min, d_max]
        all_summaries.append(summary)

        summary_path = log_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n  Band [{d_min:.2g}, {d_max:.2g}]  "
              f"baselines={summary['total_baseline_runs']}  "
              f"perturbations={summary['total_perturbation_runs']}  "
              f"cells={summary['total_cells']} ({summary['total_undersampled_cells']} undersampled)  "
              f"warnings={summary['total_warnings']}  "
              f"time={summary['total_wall_time']:.1f}s  "
              f"logs={log_dir}/")

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    if len(all_summaries) > 1:
        total_pert = sum(s["total_perturbation_runs"] for s in all_summaries)
        total_time = sum(s["total_wall_time"] for s in all_summaries)
        print(f"  Bands:         {len(all_summaries)}")
        print(f"  Perturbations: {total_pert} (across all bands)")
        print(f"  Wall time:     {total_time:.1f}s")
    else:
        s = all_summaries[0]
        print(f"  Baselines:     {s['total_baseline_runs']}")
        print(f"  Perturbations: {s['total_perturbation_runs']}")
        print(f"  Cells:         {s['total_cells']} ({s['total_undersampled_cells']} undersampled)")
        print(f"  Warnings:      {s['total_warnings']}")
        print(f"  Wall time:     {s['total_wall_time']:.1f}s")
    print(f"  Logs:          {root_log_dir}/")


if __name__ == "__main__":
    main()
