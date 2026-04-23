#!/usr/bin/env python3
"""
recover_travel_captures.py
==========================

Recovery script for travel suite captures missing from profiling_logs_v7.

Root cause: the travel payload contains "it's".  In the rendered token stream
it appears as "it\\'s" (Python repr/backslash escaping, not YAML '' doubling).
find_injection_token_span's Try 3 only handled YAML escaping, so the
attack_payload span was never found and capture_perturbed_attention returned
None for all 2275 travel perturbation runs.

Try 4 (backslash-unescape) is now in find_injection_token_span and fixes future
runs.  This script back-fills the existing v7 data without re-running AgentDojo:

  For each travel perturbation run that has no capture:
    1. Load parent baseline's token_ids, spans, reasoning_positions.
    2. Locate attack_payload span via find_injection_token_span (now has Try 4).
    3. Tokenize the stored perturbed attack_string.
    4. Call capture_perturbed_attention → one GPU forward pass per run.
    5. Append a new profiling_capture event to the same DB.

Only appends — never modifies or deletes existing rows.
Idempotent: already-captured runs are skipped.

Usage:
    cd profiler
    python recover_travel_captures.py \\
        --log-dir profiling_logs_v7 \\
        --model openai/gpt-oss-20b \\
        --device cuda:0
"""

import argparse
import json
import logging
import sys
import types
from pathlib import Path

import dill
import torch

sys.path.append("..")

from utils.experiment_logger_sql import ExperimentLogger
from profiling_logger import ProfilingLogger
from inference_runner import LocalHarmonyLLM
from feature_hooks import FeatureConfig
from profiling import capture_perturbed_attention
from perturbation_profiling import _log_first_token_tensors

logger = logging.getLogger(__name__)


def _load_db(log_dir: Path):
    import sqlite3
    conn = sqlite3.connect(log_dir / "experiment_logs.db")
    cur = conn.cursor()

    cur.execute("SELECT metadata_json, object_data FROM logs WHERE event='profiling_run'")
    runs = {}
    run_payloads = {}
    for meta_json, obj_data in cur.fetchall():
        m = json.loads(meta_json)
        rid = m["run_id"]
        runs[rid] = m
        run_payloads[rid] = str(dill.loads(obj_data))

    cur.execute("SELECT metadata_json, object_data FROM logs WHERE event='profiling_capture'")
    cap_by_run = {}
    for meta_json, obj_data in cur.fetchall():
        rid = json.loads(meta_json)["run_id"]
        cap_by_run[rid] = dill.loads(obj_data)

    conn.close()
    return runs, run_payloads, cap_by_run


def _find_payload_span(formatter, token_ids, reasoning_positions, original_payload):
    """Locate attack_payload span in the prompt portion of token_ids.

    Searches only the prompt region (everything before reasoning_positions[0])
    to avoid false matches in the generated reasoning text.

    Returns a synthetic span dict or None.
    """
    prompt_end = reasoning_positions[0] if reasoning_positions else len(token_ids)
    result = formatter.find_injection_token_span(
        token_ids, 0, prompt_end, original_payload
    )
    if result is None:
        return None
    tok_s, tok_e = result
    return {
        "tag": "attack_payload",
        "start": tok_s,
        "end": tok_e,
        "msg_idx": -1, "step": 0, "role": "tool", "channel": None,
    }


def recover(log_dir: Path, llm: LocalHarmonyLLM, dry_run: bool = False):
    logger.info("Loading DB from %s", log_dir)
    runs, run_payloads, cap_by_run = _load_db(log_dir)

    # Find travel perturbation runs missing a capture
    missing = [
        m for rid, m in runs.items()
        if m.get("suite") == "travel"
        and m.get("source") == "perturbation"
        and rid not in cap_by_run
    ]
    logger.info("Travel perturbation runs missing captures: %d", len(missing))

    if not missing:
        logger.info("Nothing to recover.")
        return

    # Group by parent_run_id so we load each baseline cap once
    from collections import defaultdict
    by_parent = defaultdict(list)
    for m in missing:
        by_parent[m["parent_run_id"]].append(m)

    logger.info("Distinct baseline parents: %d", len(by_parent))

    # Open the logger for appending
    exp_logger = ExperimentLogger(str(log_dir))
    prof_logger = ProfilingLogger(exp_logger, tensor_dir=str(log_dir / "tensors"))

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for parent_id, children in by_parent.items():
        # Load baseline capture
        bl_cap_obj = cap_by_run.get(parent_id)
        if bl_cap_obj is None:
            logger.warning("  Parent %s has no capture — skipping %d children",
                           parent_id, len(children))
            n_fail += len(children)
            continue

        bl_token_ids = bl_cap_obj.get("token_ids")
        bl_spans     = bl_cap_obj.get("spans", [])
        bl_rpos      = bl_cap_obj.get("reasoning_positions", [])

        if not bl_token_ids or not bl_rpos:
            logger.warning("  Parent %s: missing token_ids or reasoning_positions", parent_id)
            n_fail += len(children)
            continue

        # Locate attack_payload in baseline token sequence.
        # The baseline spans have no attack_payload tag (that's the bug we're fixing),
        # so we search the token stream directly using the fixed find_injection_token_span.
        original_payload = run_payloads.get(parent_id)
        if not original_payload:
            logger.warning("  Parent %s: no payload string found", parent_id)
            n_fail += len(children)
            continue

        payload_span = _find_payload_span(
            llm.formatter, bl_token_ids, bl_rpos, original_payload
        )
        if payload_span is None:
            logger.warning(
                "  Parent %s: could not locate attack_payload span (payload=%r...)",
                parent_id, original_payload[:60]
            )
            n_fail += len(children)
            continue

        # Augment baseline spans with the synthetic attack_payload span
        bl_spans_with_payload = list(bl_spans) + [payload_span]

        logger.info("  Parent %s: payload span [%d, %d), %d children",
                    parent_id, payload_span["start"], payload_span["end"], len(children))

        for m in children:
            rid = m["run_id"]
            perturbed_payload = run_payloads.get(rid, "")

            perturbed_ids = llm.tokenizer.encode(perturbed_payload, add_special_tokens=False)

            if dry_run:
                logger.info("    [dry-run] would capture run %s", rid)
                n_ok += 1
                continue

            try:
                cap_result = capture_perturbed_attention(
                    llm.model,
                    bl_token_ids,
                    bl_spans_with_payload,
                    bl_rpos,
                    perturbed_ids,
                    llm.device,
                    llm.feature_config,
                    original_payload_ids=llm.tokenizer.encode(
                        original_payload, add_special_tokens=False
                    ),
                )
            except Exception as e:
                logger.error("    Run %s: capture failed: %s", rid, e)
                n_fail += 1
                continue

            if cap_result is None:
                logger.warning("    Run %s: capture_perturbed_attention returned None", rid)
                n_fail += 1
                continue

            p_cap, p_spans, p_rpos = cap_result

            pert_out = {
                "cap":                 p_cap,
                "tagged":              types.SimpleNamespace(spans=p_spans),
                "reasoning_positions": p_rpos,
                "call_idx":            m.get("call_idx"),
            }
            _log_first_token_tensors(prof_logger, rid, pert_out)
            n_ok += 1

            if n_ok % 100 == 0:
                logger.info("    Progress: %d captured, %d failed so far", n_ok, n_fail)

    logger.info("Recovery complete: %d captured, %d skipped, %d failed",
                n_ok, n_skip, n_fail)


def main():
    p = argparse.ArgumentParser(description="Recover missing travel captures in a profiling log dir")
    p.add_argument("--log-dir", required=True, help="Path to profiling_logs_vN directory")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without running any forward passes")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["transformers", "torch", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    log_dir = Path(args.log_dir)
    if not (log_dir / "experiment_logs.db").exists():
        logger.error("No experiment_logs.db found in %s", log_dir)
        sys.exit(1)

    logger.info("Loading model %s on %s", args.model, args.device)
    llm = LocalHarmonyLLM(
        model_name=args.model,
        device=args.device,
        reasoning_effort="low",
    )
    n_layers = len(llm.feature_capture._layer_modules)
    llm.feature_config = FeatureConfig(
        layers=list(range(n_layers)),
        capture_attention=True,
        capture_hidden_states=False,
        attention_last_row_only=False,
    )
    logger.info("Model loaded: %d layers", n_layers)

    recover(log_dir, llm, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
