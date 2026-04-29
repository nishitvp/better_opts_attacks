"""
perturbation_profiling.py
=========================

Perturbation-based profiling for prompt injection analysis.

Generates systematic perturbations of successful attack payloads,
filters by distance metrics, runs through full AgentDojo pipeline,
and profiles attention/hidden-state features at each post-injection
inference step (including reasoning tokens).

Everything is plain functions operating on dicts. Distance functions
and perturbation generators are pluggable via registries.

Usage:
    from perturbation_profiling import run_perturbation_experiment

    inputs = [
        {"suite": "workspace", "user_task_id": "user_task_0",
         "injection_task_id": "injection_task_0",
         "payload": "Ignore previous instructions..."},
    ]

    config = {
        "perturbation_types": ["delete", "flip"],
        "positions": ["beginning", "end", "random_middle"],
        "n_values": [20, 40, 60, 80, 100],
        "target_per_cell": 32,
        "max_candidates_per_cell": 256,
        "distance_fn_name": "semantic",
        "distance_range": [0.1, 0.3],
    }

    summary = run_perturbation_experiment(llm, pipeline, inputs, config, prof_logger)
"""

from __future__ import annotations

import logging
import time
import random
import math
import traceback
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from harmony_format import HARMONY_SPECIAL_IDS
from inference_runner import LocalHarmonyLLM, build_pipeline, run_task, TaskResult
from profiling import (
    tag_task_result,
    tag_step_with_reasoning,
    get_all_call_indices_after_injection,
    capture_reasoning_entry_attention,
    capture_perturbed_attention,
    capture_forward_pass,
    compute_attention_mass,
    compute_scalar_features,
    save_capture,
)
from feature_hooks import FeatureConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Grid
    "perturbation_types": ["delete", "flip"],
    "positions": ["beginning", "end", "random_middle"],
    "n_values": [20, 40, 60, 80, 100, 120, 140, 160],

    # Filtering
    "target_per_cell": 32,
    "max_candidates_per_cell": 256,
    "distance_fn_name": "semantic",       # key into DISTANCE_FUNCTIONS
    "distance_range": [0.1, 0.3],         # [min, max] inclusive

    # Profiling
    "profile_from_injection": True,       # capture at all post-injection steps
    "save_tensors": False,                # save raw attn/hidden tensors to disk
    "profile_fine_groups": True,          # log fine-grained span groups too
    "n_reasoning_tokens": 10,             # how many first reasoning tokens to capture

    # Model
    "seed": 42,
    "agentdojo_version": "v1.2",

    # Payload length safety
    "max_n_fraction": 0.85,              # skip cells where N > this * payload_len

    # Failure tree: grow a second level of perturbations from diverse level-1
    # failures to address class imbalance and cover distant regions of token space.
    # Set failure_tree_n_roots > 0 to enable; 0 = disabled.
    "failure_tree_n_roots": 0,            # how many failure seeds to branch from
    "failure_tree_n_values": [1, 2, 4],   # N values for tree children (small = local)
    "failure_tree_target_per_cell": 16,   # children per (seed, N, position)
}


def _resolve_config(user_config: Optional[dict]) -> dict:
    """Merge user config over defaults, validate."""
    cfg = dict(DEFAULT_CONFIG)
    if user_config:
        cfg.update(user_config)

    # Validate
    assert cfg["target_per_cell"] > 0, "target_per_cell must be > 0"
    assert cfg["max_candidates_per_cell"] >= cfg["target_per_cell"], \
        "max_candidates_per_cell must be >= target_per_cell"
    dr = cfg["distance_range"]
    assert len(dr) == 2 and dr[0] <= dr[1], \
        f"distance_range must be [min, max] with min<=max, got {dr}"
    assert cfg["distance_fn_name"] in DISTANCE_FUNCTIONS, \
        f"Unknown distance function: {cfg['distance_fn_name']}. " \
        f"Available: {list(DISTANCE_FUNCTIONS.keys())}"

    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Protected position detection
# ═══════════════════════════════════════════════════════════════════════

def find_protected_positions(tokenizer, token_ids: List[int], vital_strings: List[str]) -> set:
    """Return token indices that overlap any of the given vital strings.

    vital_strings: exact substrings of the payload that must not be
    perturbed — e.g. account numbers, email addresses, specific names.
    Specify these per-injection in the input dict under "vital_strings".

    Maps each vital string to character spans in the decoded payload text,
    then maps those spans to token indices via prefix-decoding. A token is
    protected if any part of it overlaps a vital span.
    """
    if not token_ids or not vital_strings:
        return set()

    full_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    vital_spans: List[Tuple[int, int]] = []
    for s in vital_strings:
        start = 0
        while True:
            idx = full_text.find(s, start)
            if idx == -1:
                break
            vital_spans.append((idx, idx + len(s)))
            start = idx + 1

    if not vital_spans:
        return set()

    # Map to token positions via prefix-decoding to handle subword tokenization.
    protected: set = set()
    prev_len = 0
    for i in range(len(token_ids)):
        cur_len = len(tokenizer.decode(token_ids[:i + 1], skip_special_tokens=True))
        tok_start, tok_end = prev_len, cur_len
        for vstart, vend in vital_spans:
            if tok_start < vend and tok_end > vstart:
                protected.add(i)
                break
        prev_len = cur_len

    return protected


# ═══════════════════════════════════════════════════════════════════════
# Perturbation generators
# ═══════════════════════════════════════════════════════════════════════

def _resolve_position(position: str, N: int, seq_len: int, rng: random.Random) -> int:
    """Resolve a position type to a concrete start index."""
    if position == "beginning":
        return 0
    elif position == "end":
        return seq_len - N
    elif position == "random_middle":
        # At least 1 token margin from both ends so it's truly "middle"
        lo = max(1, 1)
        hi = max(lo, seq_len - N - 1)
        return rng.randint(lo, hi)
    else:
        raise ValueError(f"Unknown position type: {position}")


def perturb_delete(
    token_ids: List[int],
    N: int,
    position: str,
    rng: random.Random,
    protected_positions: Optional[set] = None,
) -> dict:
    """Delete N contiguous tokens from token_ids.

    If protected_positions is given, raises ValueError when the deletion
    window overlaps any protected index. The caller (generate_and_filter)
    will retry with a new random window.

    Returns a perturbation dict (not a class).
    """
    seq_len = len(token_ids)
    start = _resolve_position(position, N, seq_len, rng)
    end = start + N

    if protected_positions:
        overlap = set(range(start, end)) & protected_positions
        if overlap:
            raise ValueError(
                f"delete window [{start}, {end}) overlaps protected positions {overlap}"
            )

    perturbed = token_ids[:start] + token_ids[end:]

    return {
        "perturbed_ids": perturbed,
        "original_ids": list(token_ids),
        "perturbation_type": "delete",
        "N": N,
        "position": position,
        "position_start": start,
        "position_end": end,
        "original_len": seq_len,
        "perturbed_len": len(perturbed),
        "deleted_ids": token_ids[start:end],
    }


def perturb_flip(
    token_ids: List[int],
    N: int,
    position: str,
    vocab_size: int,
    special_ids: set,
    rng: random.Random,
    protected_positions: Optional[set] = None,
) -> dict:
    """Flip up to N contiguous tokens to random non-special tokens.

    Protected positions within the window are left unchanged. If fewer
    than N positions are available (all others are protected), the actual
    flip count is less than N — recorded in "n_flipped". This avoids
    trivial failures where a changed account number or email address causes
    the injection to fail for reasons unrelated to its structural properties.

    Returns a perturbation dict.
    """
    seq_len = len(token_ids)
    start = _resolve_position(position, N, seq_len, rng)
    end = start + N

    perturbed = list(token_ids)
    original_segment = token_ids[start:end]
    flipped_to = []
    n_flipped = 0

    for i in range(start, end):
        if protected_positions and i in protected_positions:
            flipped_to.append(token_ids[i])  # leave unchanged
            continue
        # Sample until we get a non-special, different token
        for _ in range(100):
            new_id = rng.randint(0, vocab_size - 1)
            if new_id not in special_ids and new_id != token_ids[i]:
                break
        perturbed[i] = new_id
        flipped_to.append(new_id)
        n_flipped += 1

    return {
        "perturbed_ids": perturbed,
        "original_ids": list(token_ids),
        "perturbation_type": "flip",
        "N": N,
        "n_flipped": n_flipped,
        "position": position,
        "position_start": start,
        "position_end": end,
        "original_len": seq_len,
        "perturbed_len": len(perturbed),
        "original_segment": original_segment,
        "flipped_to": flipped_to,
    }


PERTURBATION_GENERATORS = {
    "delete": perturb_delete,
    "flip": perturb_flip,
}


def generate_perturbation(
    token_ids: List[int],
    perturbation_type: str,
    N: int,
    position: str,
    vocab_size: int,
    special_ids: set,
    rng: random.Random,
    protected_positions: Optional[set] = None,
) -> dict:
    """Dispatch to the appropriate generator."""
    if perturbation_type == "delete":
        return perturb_delete(token_ids, N, position, rng, protected_positions)
    elif perturbation_type == "flip":
        return perturb_flip(token_ids, N, position, vocab_size, special_ids, rng, protected_positions)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


# ═══════════════════════════════════════════════════════════════════════
# Sanity checks
# ═══════════════════════════════════════════════════════════════════════

def check_perturbation(pert: dict, special_ids: set) -> List[str]:
    """Validate a perturbation dict. Returns list of warnings (empty = OK)."""
    warnings = []
    ptype = pert["perturbation_type"]
    N = pert["N"]
    orig_len = pert["original_len"]
    pert_len = pert["perturbed_len"]
    pert_ids = pert["perturbed_ids"]

    if ptype == "delete":
        expected_len = orig_len - N
        if pert_len != expected_len:
            warnings.append(
                f"delete: expected len {expected_len}, got {pert_len}")

    elif ptype == "flip":
        if pert_len != orig_len:
            warnings.append(
                f"flip: expected same len {orig_len}, got {pert_len}")

        # Count actual differences — when protected positions are present,
        # n_flipped < N is expected, so use the stored count if available.
        orig_ids = pert["original_ids"]
        diffs = sum(1 for a, b in zip(orig_ids, pert_ids) if a != b)
        expected_diffs = pert.get("n_flipped", N)
        if diffs != expected_diffs:
            warnings.append(
                f"flip: expected {expected_diffs} diffs, got {diffs}")

        # Check no special tokens introduced
        start, end = pert["position_start"], pert["position_end"]
        for i in range(start, min(end, len(pert_ids))):
            if pert_ids[i] in special_ids:
                warnings.append(
                    f"flip: special token {pert_ids[i]} at position {i}")
                break

    # Position bounds
    if pert["position_start"] < 0:
        warnings.append(f"position_start < 0: {pert['position_start']}")
    if pert["position_end"] > orig_len:
        warnings.append(f"position_end > original length: "
                        f"{pert['position_end']} > {orig_len}")

    return warnings


def check_task_result(result: TaskResult, label: str) -> List[str]:
    """Validate a TaskResult after pipeline execution."""
    warnings = []

    if not result.messages:
        warnings.append(f"{label}: no messages")
    else:
        roles = [m["role"] for m in result.messages]
        if "assistant" not in roles:
            warnings.append(f"{label}: no assistant messages")

    if not result.evaluation:
        warnings.append(f"{label}: empty evaluation")
    else:
        for key in ["utility", "security"]:
            if key not in result.evaluation:
                warnings.append(f"{label}: missing evaluation key '{key}'")

    if not result.call_log:
        warnings.append(f"{label}: empty call_log")
    else:
        for ci, entry in enumerate(result.call_log):
            if "prompt_token_ids" not in entry:
                warnings.append(f"{label}: call_log[{ci}] missing prompt_token_ids")
            if "gen_token_ids" not in entry:
                warnings.append(f"{label}: call_log[{ci}] missing gen_token_ids")

    if result.runtime is None:
        warnings.append(f"{label}: runtime is None")

    return warnings


def check_distance(value: float, fn_name: str) -> List[str]:
    """Validate a distance value."""
    warnings = []
    if not math.isfinite(value):
        warnings.append(f"distance ({fn_name}): non-finite value {value}")
    if value < 0:
        warnings.append(f"distance ({fn_name}): negative value {value}")
    return warnings


def check_capture(capture: dict, expected_layers: List[int], label: str) -> List[str]:
    """Validate a capture dict."""
    warnings = []

    if not capture.get("attention"):
        warnings.append(f"{label}: no attention maps captured")
    else:
        for layer_idx in expected_layers:
            if layer_idx not in capture["attention"]:
                warnings.append(f"{label}: missing attention at layer {layer_idx}")

    if capture.get("seq_len", 0) == 0:
        warnings.append(f"{label}: seq_len is 0")

    return warnings


def check_tagged(tagged, full_len: int, label: str) -> List[str]:
    """Validate a TaggedConversation."""
    warnings = []

    if tagged.seq_len != full_len:
        warnings.append(
            f"{label}: tagged seq_len {tagged.seq_len} != expected {full_len}")

    # Check coverage
    covered = torch.zeros(tagged.seq_len, dtype=torch.bool)
    for tag, mask in tagged.masks.items():
        covered |= mask
    uncovered = int((~covered).sum())
    if uncovered > 0:
        warnings.append(f"{label}: {uncovered} uncovered tokens")

    if not tagged.spans:
        warnings.append(f"{label}: no spans")

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Distance functions
# ═══════════════════════════════════════════════════════════════════════
#
# Signature:  (model, tokenizer, device, original_ids, perturbed_ids) -> float
#
# Higher value = more different. Add new ones by inserting into
# DISTANCE_FUNCTIONS dict.
# ═══════════════════════════════════════════════════════════════════════

def _embed_tokens(model, device, token_ids: List[int]) -> torch.Tensor:
    """Forward pass → mean-pooled last hidden state. Returns 1-D tensor on CPU."""
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    # Last hidden layer, batch 0
    hidden = out.hidden_states[-1][0]       # (seq_len, hidden_dim)
    embedding = hidden.mean(dim=0)          # (hidden_dim,)
    return embedding.cpu()


def _token_perplexity(model, device, token_ids: List[int]) -> float:
    """Forward pass → cross-entropy loss (log-perplexity)."""
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return out.loss.item()


def semantic_distance(model, tokenizer, device, original_ids, perturbed_ids) -> float:
    """1 - cosine_similarity between mean-pooled last hidden states.

    Uses GPT-OSS's own representations so small token-level changes
    register as real shifts (unlike sentence-transformers which are
    trained to be invariant to paraphrasing).

    Returns float in [0, 2].
    """
    emb_orig = _embed_tokens(model, device, original_ids)
    emb_pert = _embed_tokens(model, device, perturbed_ids)
    cos = F.cosine_similarity(emb_orig.unsqueeze(0), emb_pert.unsqueeze(0)).item()
    return 1.0 - cos


def fluency_distance(model, tokenizer, device, original_ids, perturbed_ids) -> float:
    """Increase in cross-entropy loss (nats/token) of perturbed vs original.

    distance = max(0, loss_perturbed - loss_original)

    Working in log-space avoids the ratio formula exploding when random-token
    flips push perplexity into the thousands: exp(loss_pert) can be enormous
    but loss_pert - loss_orig is bounded to a few nats even for heavy
    perturbations.

    Typical values:
        ~0.0  — nearly identical sequences
        ~1–3  — moderate perturbation (20–40% of tokens flipped)
        ~5–8  — heavy perturbation (most tokens random)

    Returns float >= 0.
    """
    loss_orig = _token_perplexity(model, device, original_ids)
    loss_pert = _token_perplexity(model, device, perturbed_ids)
    return max(0.0, loss_pert - loss_orig)


DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    "semantic": semantic_distance,
    "fluency": fluency_distance,
}


def bind_distance_fn(
    fn_name: str,
    model,
    tokenizer,
    device: str,
    original_ids: List[int],
) -> Callable[[List[int]], float]:
    """Precompute the original side of a distance function.

    Returns a closure  dist(perturbed_ids) -> float  that reuses the cached
    original representation. Call this once per payload, then pass the
    returned closure to generate_and_filter for all grid cells.
    """
    if fn_name == "semantic":
        orig_emb = _embed_tokens(model, device, original_ids)

        def _semantic(perturbed_ids: List[int]) -> float:
            pert_emb = _embed_tokens(model, device, perturbed_ids)
            cos = F.cosine_similarity(orig_emb.unsqueeze(0), pert_emb.unsqueeze(0)).item()
            return 1.0 - cos

        return _semantic

    elif fn_name == "fluency":
        orig_loss = _token_perplexity(model, device, original_ids)

        def _fluency(perturbed_ids: List[int]) -> float:
            pert_loss = _token_perplexity(model, device, perturbed_ids)
            return max(0.0, pert_loss - orig_loss)

        return _fluency

    else:
        # Fallback: wrap the full function (no caching)
        fn = DISTANCE_FUNCTIONS[fn_name]

        def _fallback(perturbed_ids: List[int]) -> float:
            return fn(model, tokenizer, device, original_ids, perturbed_ids)

        return _fallback

# Per-metric default distance ranges.
# Semantic distance (1 − cosine sim) is in [0, 1] and reasonably uniform;
# [0.1, 0.3] targets mild-to-moderate token-level changes.
# Fluency distance is loss_pert - loss_orig in nats/token (log-space diff).
# [0.5, 4.0] targets perturbations that noticeably hurt fluency but aren't
# completely degenerate. Typical: ~1 nat = 20–40% tokens flipped, ~4 nats =
# heavily perturbed but payload structure still somewhat present.
DISTANCE_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "semantic": (0.1, 0.3),
    "fluency":  (0.5, 4.0),
}


# ═══════════════════════════════════════════════════════════════════════
# Payload tokenization helpers
# ═══════════════════════════════════════════════════════════════════════

def tokenize_payload(tokenizer, payload: str) -> List[int]:
    """Tokenize a payload string, no special tokens."""
    return tokenizer.encode(payload, add_special_tokens=False)


def decode_payload(tokenizer, token_ids: List[int]) -> str:
    """Decode token IDs back to a payload string."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def compute_prefix_kv_cache(model, prefix_ids: List[int], device: str):
    """Run a single forward pass on prefix_ids and return past_key_values.

    The returned object is stored (on CPU) in LocalHarmonyLLM.set_prefix_cache()
    and reused across all perturbation runs in the same context to skip re-prefilling
    the shared prompt prefix on every model.generate() call.
    """
    import torch as _torch
    input_tensor = _torch.tensor([prefix_ids], device=device)
    with _torch.no_grad():
        out = model(input_ids=input_tensor, use_cache=True)
    return out.past_key_values


# ═══════════════════════════════════════════════════════════════════════
# Cell-level: generate, filter, return surviving perturbations
# ═══════════════════════════════════════════════════════════════════════

def generate_and_filter(
    original_ids: List[int],
    perturbation_type: str,
    N: int,
    position: str,
    bound_distance_fn: Callable[[List[int]], float],
    distance_fn_name: str,
    distance_range: Tuple[float, float],
    target_count: int,
    max_candidates: int,
    vocab_size: int,
    special_ids: set,
    rng: random.Random,
    protected_positions: Optional[set] = None,
) -> dict:
    """Generate perturbations, measure distance, filter by range.

    bound_distance_fn is a closure returned by bind_distance_fn — the original
    side is already precomputed, so only the perturbed sequence needs to be
    evaluated per candidate.

    Returns a dict with:
        survivors: list of perturbation dicts (with distance added)
        stats: {generated, passed_sanity, in_range, target, ...}
        all_distances: list of all measured distances (for diagnostics)
    """
    d_min, d_max = distance_range
    survivors = []
    all_distances = []
    generated = 0
    sanity_failed = 0
    out_of_range = 0

    cell_label = f"{perturbation_type}/N={N}/{position}"

    for attempt in range(max_candidates):
        if len(survivors) >= target_count:
            break

        # ── Generate ────────────────────────────────────────────────
        try:
            pert = generate_perturbation(
                original_ids, perturbation_type, N, position,
                vocab_size, special_ids, rng,
                protected_positions=protected_positions,
            )
        except Exception as e:
            logger.warning("  %s: generation failed (attempt %d): %s",
                           cell_label, attempt, e)
            continue
        generated += 1

        # ── Sanity check ────────────────────────────────────────────
        warnings = check_perturbation(pert, special_ids)
        if warnings:
            sanity_failed += 1
            for w in warnings:
                logger.warning("  %s: sanity: %s", cell_label, w)
            continue

        # ── Measure distance ────────────────────────────────────────
        try:
            dist = bound_distance_fn(pert["perturbed_ids"])
        except Exception as e:
            logger.warning("  %s: distance computation failed: %s",
                           cell_label, e)
            continue

        dist_warnings = check_distance(dist, distance_fn_name)
        if dist_warnings:
            for w in dist_warnings:
                logger.warning("  %s: %s", cell_label, w)
            continue

        all_distances.append(dist)
        pert["distance"] = dist
        pert["distance_fn_name"] = distance_fn_name

        # ── Filter ──────────────────────────────────────────────────
        if d_min <= dist <= d_max:
            survivors.append(pert)
        else:
            out_of_range += 1

    undersampled = len(survivors) < target_count

    stats = {
        "cell": cell_label,
        "perturbation_type": perturbation_type,
        "N": N,
        "position": position,
        "generated": generated,
        "sanity_failed": sanity_failed,
        "in_range": len(survivors),
        "out_of_range": out_of_range,
        "target": target_count,
        "max_candidates": max_candidates,
        "undersampled": undersampled,
        "distance_fn_name": distance_fn_name,
        "distance_range": list(distance_range),
    }

    if all_distances:
        stats["distance_min"] = min(all_distances)
        stats["distance_max"] = max(all_distances)
        stats["distance_mean"] = sum(all_distances) / len(all_distances)

    if undersampled:
        logger.warning(
            "  %s: undersampled — got %d/%d after %d attempts "
            "(dist range [%.3f, %.3f])",
            cell_label, len(survivors), target_count, generated,
            d_min, d_max,
        )

    return {"survivors": survivors, "stats": stats, "all_distances": all_distances}


# ═══════════════════════════════════════════════════════════════════════
# First-token profile: run pipeline, capture prompt-only attention
# ═══════════════════════════════════════════════════════════════════════

def profile_first_token_only(
    llm: LocalHarmonyLLM,
    pipeline,
    suite_name: str,
    user_task_id: str,
    injection_task_id: str,
    payload_string: str,
    feature_config: FeatureConfig,
    *,
    attack_payload_for_tagging: Optional[str] = None,
    n_reasoning_tokens: int = 10,
    run_label: str = "run",
    version: str = "v1.2",
) -> dict:
    """Run full pipeline, then capture raw attention/hidden-state tensors at
    the first N reasoning tokens.

    No analysis is done here — tensors are returned as-is for offline study.

    Returns:
        {
            task_result,
            call_idx,             # which call_log entry was captured
            cap,                  # raw capture dict {attention, hidden_states, token_ids, seq_len}
            tagged,               # TaggedConversation for the prompt portion
            reasoning_positions,  # absolute token indices that were captured
            warnings,
            wall_time,
        }
    """
    all_warnings = []
    t0 = time.time()

    # Re-arm the prefix cache for this call (no-op if no cache is set).
    # Safety guard: ensures this function can be called at any point in
    # the pipeline flow without inheriting a stale disarmed state.
    llm.rearm_prefix_cache()

    # ── Run full pipeline ───────────────────────────────────────────
    try:
        result = run_task(
            pipeline, llm, suite_name, user_task_id,
            injection_task_id, payload_string,
            version=version,
        )
    except Exception as e:
        logger.error("  %s: pipeline failed: %s", run_label, e)
        return {
            "task_result": None,
            "warnings": [f"pipeline_failed: {e}"],
            "wall_time": time.time() - t0,
            "t_pipeline": time.time() - t0,
            "t_capture": 0.0,
            "error": str(e),
        }

    t_pipeline = time.time() - t0
    result_warnings = check_task_result(result, run_label)
    all_warnings.extend(result_warnings)

    # ── Capture reasoning-entry attention ───────────────────────────
    t_cap0 = time.time()
    tagging_payload = attack_payload_for_tagging or payload_string
    cap_result = capture_reasoning_entry_attention(
        llm.model, llm.formatter, result, llm.device, feature_config,
        attack_payload=tagging_payload,
        n_reasoning_tokens=n_reasoning_tokens,
    )
    t_capture = time.time() - t_cap0

    logger.info("  %s: t_pipeline=%.1fs  t_capture=%.1fs  total=%.1fs",
                run_label, t_pipeline, t_capture, t_pipeline + t_capture)

    if cap_result is None:
        all_warnings.append(f"{run_label}: injection not found or no reasoning tokens")
        return {
            "task_result": result,
            "call_idx": None,
            "cap": None,
            "tagged": None,
            "reasoning_positions": [],
            "warnings": all_warnings,
            "wall_time": t_pipeline + t_capture,
            "t_pipeline": t_pipeline,
            "t_capture": t_capture,
        }

    cap, tagged, call_idx, reasoning_positions = cap_result

    cap_warnings = check_capture(cap, feature_config.layers or [], run_label)
    all_warnings.extend(cap_warnings)

    return {
        "task_result": result,
        "call_idx": call_idx,
        "cap": cap,
        "tagged": tagged,
        "reasoning_positions": reasoning_positions,
        "warnings": all_warnings,
        "wall_time": t_pipeline + t_capture,
        "t_pipeline": t_pipeline,
        "t_capture": t_capture,
    }


# ═══════════════════════════════════════════════════════════════════════
# Single-run profiling: pipeline + feature extraction
# ═══════════════════════════════════════════════════════════════════════

def profile_single_run(
    llm: LocalHarmonyLLM,
    pipeline,
    suite_name: str,
    user_task_id: str,
    injection_task_id: str,
    payload_string: str,
    feature_config: FeatureConfig,
    *,
    attack_payload_for_tagging: Optional[str] = None,
    profile_from_injection: bool = True,
    profile_fine_groups: bool = True,
    save_tensors: bool = False,
    tensor_dir: Optional[str] = None,
    run_label: str = "run",
    version: str = "v1.2",
) -> dict:
    """Run pipeline + capture features. Returns a dict with everything.

    The returned dict contains:
        task_result:   TaskResult object
        steps:         list of per-step dicts, each with:
            call_idx, tagged, capture, coarse_features, fine_features,
            coarse_attention_mass, fine_attention_mass, tensor_paths
        warnings:      list of warning strings
        wall_time:     total time
    """
    all_warnings = []
    t0 = time.time()

    # Re-arm the prefix cache for this call (no-op if no cache is set).
    llm.rearm_prefix_cache()

    # ── Run pipeline ────────────────────────────────────────────────
    try:
        result = run_task(
            pipeline, llm, suite_name, user_task_id,
            injection_task_id, payload_string,
            version=version,
        )
    except Exception as e:
        logger.error("  %s: pipeline failed: %s", run_label, e)
        return {
            "task_result": None,
            "steps": [],
            "warnings": [f"pipeline_failed: {e}"],
            "wall_time": time.time() - t0,
            "error": str(e),
        }

    # ── Validate result ─────────────────────────────────────────────
    result_warnings = check_task_result(result, run_label)
    all_warnings.extend(result_warnings)
    for w in result_warnings:
        logger.warning("  %s", w)

    if result.call_log is None or len(result.call_log) == 0:
        all_warnings.append(f"{run_label}: no call_log, skipping profiling")
        return {
            "task_result": result,
            "steps": [],
            "warnings": all_warnings,
            "wall_time": time.time() - t0,
        }

    # ── Find post-injection steps ───────────────────────────────────
    tagging_payload = attack_payload_for_tagging or payload_string

    if profile_from_injection:
        post_inj_indices = get_all_call_indices_after_injection(
            result, attack_payload=tagging_payload,
        )
        if not post_inj_indices:
            all_warnings.append(f"{run_label}: no post-injection steps found")
            # Fall back to profiling all steps
            post_inj_indices = list(range(len(result.call_log)))
    else:
        post_inj_indices = list(range(len(result.call_log)))

    # ── Profile each step ───────────────────────────────────────────
    step_results = []
    expected_layers = feature_config.layers or []

    for call_idx in post_inj_indices:
        step_label = f"{run_label}/step={call_idx}"

        try:
            # Tag with reasoning
            full_ids, tagged = tag_step_with_reasoning(
                llm.formatter, result, call_idx,
                attack_payload=tagging_payload,
            )

            # Validate tagging
            tag_warnings = check_tagged(tagged, len(full_ids), step_label)
            all_warnings.extend(tag_warnings)

            # Capture forward pass
            cap = capture_forward_pass(
                llm.model, full_ids, llm.device, feature_config,
            )

            # Validate capture
            cap_warnings = check_capture(cap, expected_layers, step_label)
            all_warnings.extend(cap_warnings)

            # Compute features with coarse groups
            coarse_groups = tagged.to_span_groups(fine=False)
            coarse_mass = compute_attention_mass(cap, coarse_groups, len(full_ids))
            coarse_feats = compute_scalar_features(cap, coarse_groups, len(full_ids))

            # Compute features with fine groups
            fine_mass = {}
            fine_feats = []
            if profile_fine_groups:
                fine_groups = tagged.to_span_groups(fine=True)
                fine_mass = compute_attention_mass(cap, fine_groups, len(full_ids))
                fine_feats = compute_scalar_features(cap, fine_groups, len(full_ids))

            # Optionally save tensors
            tensor_paths = {}
            if save_tensors and tensor_dir:
                for layer_idx, attn in cap["attention"].items():
                    path = f"{tensor_dir}/{run_label}_step{call_idx}_attn_L{layer_idx}.pt"
                    save_capture({"attention": {layer_idx: attn}}, path)
                    tensor_paths[f"attn_L{layer_idx}"] = path

            # Reasoning token stats
            reasoning_count = int(tagged.masks.get("assistant_reasoning",
                                  torch.zeros(1, dtype=torch.bool)).sum())
            tool_call_count = int(tagged.masks.get("assistant_tool_call",
                                  torch.zeros(1, dtype=torch.bool)).sum())

            step_results.append({
                "call_idx": call_idx,
                "seq_len": len(full_ids),
                "prompt_len": len(result.call_log[call_idx]["prompt_token_ids"]),
                "gen_len": len(result.call_log[call_idx]["gen_token_ids"]),
                "reasoning_tokens": reasoning_count,
                "tool_call_tokens": tool_call_count,
                "coarse_groups_present": sorted(coarse_groups.keys()),
                "fine_groups_present": sorted(fine_mass.keys()) if fine_mass else [],
                "coarse_features": coarse_feats,
                "fine_features": fine_feats,
                "coarse_attention_mass": _mass_to_serializable(coarse_mass),
                "fine_attention_mass": _mass_to_serializable(fine_mass),
                "tensor_paths": tensor_paths,
            })

        except Exception as e:
            logger.error("  %s: profiling failed: %s", step_label, e)
            traceback.print_exc()
            all_warnings.append(f"{step_label}: profiling_failed: {e}")

    return {
        "task_result": result,
        "steps": step_results,
        "warnings": all_warnings,
        "wall_time": time.time() - t0,
    }


def _mass_to_serializable(mass_dict):
    """Convert {group: {layer: tensor}} to {group: {layer: list}}."""
    out = {}
    for group, layers in mass_dict.items():
        out[group] = {}
        for layer_idx, tensor in layers.items():
            out[group][layer_idx] = tensor.tolist()
    return out


# ═══════════════════════════════════════════════════════════════════════
# Logging helpers
# ═══════════════════════════════════════════════════════════════════════

def log_baseline_run(
    prof_logger,
    result: TaskResult,
    profiling_output: dict,
    input_entry: dict,
    config: dict,
) -> str:
    """Log a baseline (unperturbed) run. Returns run_id."""

    context = {
        "suite": input_entry["suite"],
        "user_task_id": input_entry["user_task_id"],
        "injection_task_id": input_entry["injection_task_id"],
    }

    outcome = {
        "success": result.evaluation.get("injection_success"),
        "utility": result.evaluation.get("utility"),
        "security": result.evaluation.get("security"),
    }

    run_id = prof_logger.log_run(
        attack_string=input_entry["payload"],
        context=context,
        outcome=outcome,
        source="baseline",
        perturbation={"type": "none"},
        wall_time=profiling_output["wall_time"],
        n_steps_profiled=len(profiling_output["steps"]),
        n_warnings=len(profiling_output["warnings"]),
        config_distance_fn=config["distance_fn_name"],
        config_distance_range=config["distance_range"],
    )

    # Log per-step features
    _log_step_features(prof_logger, run_id, profiling_output)

    return run_id


def log_perturbation_run(
    prof_logger,
    result: TaskResult,
    profiling_output: dict,
    input_entry: dict,
    config: dict,
    parent_run_id: str,
    perturbation: dict,
    perturbed_payload: str,
) -> str:
    """Log a perturbation run. Returns run_id."""

    context = {
        "suite": input_entry["suite"],
        "user_task_id": input_entry["user_task_id"],
        "injection_task_id": input_entry["injection_task_id"],
    }

    outcome = {
        "success": result.evaluation.get("injection_success"),
        "utility": result.evaluation.get("utility"),
        "security": result.evaluation.get("security"),
    }

    pert_meta = {
        "type": perturbation["perturbation_type"],
        "N": perturbation["N"],
        "position": perturbation["position"],
        "position_start": perturbation["position_start"],
        "position_end": perturbation["position_end"],
        "original_len": perturbation["original_len"],
        "perturbed_len": perturbation["perturbed_len"],
    }

    run_id = prof_logger.log_run(
        attack_string=perturbed_payload,
        context=context,
        outcome=outcome,
        source="perturbation",
        parent_run_id=parent_run_id,
        perturbation=pert_meta,
        wall_time=profiling_output["wall_time"],
        n_steps_profiled=len(profiling_output["steps"]),
        n_warnings=len(profiling_output["warnings"]),
        config_distance_fn=config["distance_fn_name"],
        config_distance_range=config["distance_range"],
    )

    # Log distances
    prof_logger.log_perturbation_distances(run_id, parent_run_id, {
        perturbation["distance_fn_name"]: perturbation["distance"],
    })

    # Log per-step features
    _log_step_features(prof_logger, run_id, profiling_output)

    return run_id


def log_cell_stats(prof_logger, parent_run_id: str, cell_stats: dict):
    """Log stats about a grid cell (how many generated, filtered, etc.)."""
    prof_logger.logger.log(
        cell_stats,
        variable_name="cell_stats",
        event="perturbation_cell_stats",
        parent_run_id=parent_run_id,
    )


def _log_step_features(prof_logger, run_id: str, profiling_output: dict):
    """Log scalar features from all profiled steps."""
    for step_data in profiling_output["steps"]:
        call_idx = step_data["call_idx"]

        # Log coarse features
        for feat in step_data["coarse_features"]:
            prof_logger.log_scalar_features(
                run_id,
                layer=feat["layer"],
                head=feat.get("head"),
                step=call_idx,
                feature_type=feat.get("feature_type", "attention"),
                group_resolution="coarse",
                **{k: v for k, v in feat.items()
                   if k not in ("layer", "head", "feature_type")},
            )

        # Log fine features
        for feat in step_data.get("fine_features", []):
            prof_logger.log_scalar_features(
                run_id,
                layer=feat["layer"],
                head=feat.get("head"),
                step=call_idx,
                feature_type=feat.get("feature_type", "attention"),
                group_resolution="fine",
                **{k: v for k, v in feat.items()
                   if k not in ("layer", "head", "feature_type")},
            )


def _log_first_token_tensors(prof_logger, run_id: str, output: dict):
    """Save raw capture tensors to disk and log the path + metadata."""
    cap = output.get("cap")
    if cap is None:
        return

    tagged = output.get("tagged")
    reasoning_positions = output.get("reasoning_positions", [])

    # Save the full capture dict (attention + hidden_states + token_ids)
    import torch as _torch
    path = str(prof_logger.tensor_dir / f"{run_id}_cap.pt")
    _torch.save(cap, path)

    # Log path + enough metadata to reconstruct span groups offline
    prof_logger.logger.log(
        {
            "cap_path": path,
            "reasoning_positions": reasoning_positions,
            "seq_len": cap.get("seq_len"),
            "token_ids": cap.get("token_ids"),
            "spans": tagged.spans if tagged is not None else [],
        },
        variable_name="capture",
        event="profiling_capture",
        run_id=run_id,
        call_idx=output.get("call_idx"),
        capture_mode="reasoning_entry",
    )


# ═══════════════════════════════════════════════════════════════════════
# Failure tree: diverse failure generation
# ═══════════════════════════════════════════════════════════════════════

def _select_diverse_failure_seeds(
    failure_pool: List[dict],
    n_roots: int,
    rng: random.Random,
) -> List[dict]:
    """Pick n_roots diverse failures by round-robin across N-value buckets.

    Level-1 failures at small N are near the original payload; at large N
    they are far away. Sampling evenly across N buckets ensures the tree
    roots are spread across token space rather than clustered near the origin.
    """
    from collections import defaultdict
    by_n: Dict[int, List[dict]] = defaultdict(list)
    for seed in failure_pool:
        by_n[seed["N"]].append(seed)

    # Shuffle within each bucket for randomness, sort buckets by N for determinism
    buckets = []
    for n in sorted(by_n.keys()):
        bucket = by_n[n][:]
        rng.shuffle(bucket)
        buckets.append(bucket)

    selected: List[dict] = []
    while len(selected) < n_roots:
        made_progress = False
        for bucket in buckets:
            if bucket and len(selected) < n_roots:
                selected.append(bucket.pop())
                made_progress = True
        if not made_progress:
            break

    return selected


def run_failure_tree_experiment(
    llm,
    pipeline,
    seeds: List[dict],
    cfg: dict,
    prof_logger,
) -> dict:
    """Grow a multi-level tree of perturbations branching from failure seeds.

    Level 1 seeds come from failed level-1 perturbations. Each subsequent level
    seeds from the failures of the previous level, pushing progressively further
    into token space. Almost all children will also fail (we start from failures),
    which directly addresses class imbalance.

    Attention capture always reuses the ORIGINAL baseline's token structure so
    attention maps are directly comparable across all tree depths.

    Seed dict fields (initial level):
        payload, token_ids, suite, user_task_id, injection_task_id,
        parent_run_id (= level-1 perturbation run_id),
        vital_strings (optional), N,
        bl_info: {token_ids, spans, rpos} from the original baseline capture,
        original_payload_ids: token IDs of the unperturbed payload.

    Deeper levels add:
        tree_root_run_id: run_id of the level-1 perturbation that started this branch.
    """
    n_values      = cfg.get("failure_tree_n_values", [1, 2, 4])
    target_cell   = cfg.get("failure_tree_target_per_cell", 16)
    budget_decay  = cfg.get("failure_tree_budget_decay", 0.5)
    max_levels    = cfg.get("failure_tree_levels", 1)
    n_roots       = cfg.get("failure_tree_n_roots", len(seeds))
    positions     = cfg.get("positions", ["beginning", "end", "random_middle"])
    vocab_size    = llm.tokenizer.vocab_size
    special_ids   = set(HARMONY_SPECIAL_IDS)
    feature_config = llm.feature_config
    rng = random.Random(cfg.get("seed", 42) + 31337)

    tree_summary: dict = {
        "n_initial_seeds": len(seeds),
        "total_children": 0,
        "total_failures": 0,
        "total_successes": 0,
        "per_level": [],
    }

    current_seeds       = seeds
    current_target_cell = target_cell

    for level in range(max_levels):
        tree_depth   = level + 2   # depth 2 = first tree level, 3 = second, …
        next_failures: List[dict] = []
        level_stats  = {
            "level": level + 1,
            "tree_depth": tree_depth,
            "n_seeds": len(current_seeds),
            "target_per_cell": current_target_cell,
            "n_children": 0,
            "n_failures": 0,
            "n_successes": 0,
        }

        logger.info("Failure tree level %d/%d: %d seeds, target_per_cell=%d",
                    level + 1, max_levels, len(current_seeds), current_target_cell)

        for seed_idx, seed in enumerate(current_seeds):
            seed_ids = list(seed["token_ids"])
            seed_len = len(seed_ids)
            bl       = seed.get("bl_info")
            orig_ids = seed.get("original_payload_ids")

            # tree_root_run_id stays fixed to the level-1 perturbation for all depths
            tree_root_run_id = seed.get("tree_root_run_id", seed["parent_run_id"])

            vital_strings = seed.get("vital_strings") or []
            protected = (
                find_protected_positions(llm.tokenizer, seed_ids, vital_strings)
                if vital_strings else set()
            )

            context = {
                "suite":             seed["suite"],
                "user_task_id":      seed["user_task_id"],
                "injection_task_id": seed["injection_task_id"],
            }

            # Set prefix KV cache for this seed's context so that each child's
            # run_task() call can skip re-prefilling the shared prefix.
            if bl is not None:
                bl_spans_seed = bl.get("spans") or []
                bl_tids_seed  = bl.get("token_ids") or []
                payload_spans = [s for s in bl_spans_seed if s.get("tag") == "attack_payload"]
                if payload_spans and bl_tids_seed:
                    p_start = min(s["start"] for s in payload_spans)
                    prefix_ids_seed = list(bl_tids_seed[:p_start])
                    t_kv = time.time()
                    kv = compute_prefix_kv_cache(llm.model, prefix_ids_seed, llm.device)
                    llm.set_prefix_cache(prefix_ids_seed, kv)
                    logger.debug("  Tree seed %d: prefix KV cache %d tokens, %.1fs",
                                 seed_idx, p_start, time.time() - t_kv)
            else:
                llm.clear_prefix_cache()

            logger.info("  Level %d seed %d/%d  (N=%d  suite=%s/%s)",
                        level + 1, seed_idx + 1, len(current_seeds),
                        seed.get("N", "?"), seed["suite"], seed["injection_task_id"])

            for N in n_values:
                if N >= seed_len:
                    continue

                for position in positions:
                    children_generated = 0
                    attempts           = 0
                    max_attempts       = current_target_cell * 8

                    while children_generated < current_target_cell and attempts < max_attempts:
                        attempts += 1
                        t_child = time.time()

                        # ── Generate child ───────────────────────────────
                        try:
                            pert = perturb_flip(
                                seed_ids, N, position,
                                vocab_size, special_ids, rng,
                                protected_positions=protected,
                            )
                        except Exception:
                            continue

                        child_payload = decode_payload(llm.tokenizer, pert["perturbed_ids"])

                        # ── Run pipeline ─────────────────────────────────
                        # Re-arm cache so only the first post-injection generate()
                        # call in this run uses the prefix cache.
                        llm.rearm_prefix_cache()
                        try:
                            result = run_task(
                                pipeline, llm,
                                seed["suite"], seed["user_task_id"],
                                seed["injection_task_id"], child_payload,
                                version=cfg.get("agentdojo_version", "v1.2"),
                            )
                        except Exception as e:
                            logger.warning("    Tree child pipeline failed: %s", e)
                            continue

                        outcome = {
                            "success": result.evaluation.get("injection_success"),
                            "utility": result.evaluation.get("utility"),
                            "security": result.evaluation.get("security"),
                        }

                        # ── Attention capture ────────────────────────────
                        child_cap_result = None
                        if bl is not None:
                            child_cap_result = capture_perturbed_attention(
                                llm.model,
                                bl["token_ids"], bl["spans"], bl["rpos"],
                                pert["perturbed_ids"],
                                llm.device, feature_config,
                                original_payload_ids=orig_ids,
                            )

                        child_out = (
                            {
                                "cap": child_cap_result[0],
                                "tagged": types.SimpleNamespace(spans=child_cap_result[1]),
                                "reasoning_positions": child_cap_result[2],
                                "call_idx": None,
                            }
                            if child_cap_result is not None
                            else {"cap": None, "tagged": None, "reasoning_positions": [], "call_idx": None}
                        )

                        # ── Log ──────────────────────────────────────────
                        pert_meta = {
                            "type":           "flip",
                            "N":              N,
                            "n_flipped":      pert.get("n_flipped", N),
                            "position":       position,
                            "position_start": pert["position_start"],
                            "position_end":   pert["position_end"],
                            "original_len":   pert["original_len"],
                            "perturbed_len":  pert["perturbed_len"],
                            "tree_seed_N":    seed.get("N", 0),
                        }
                        child_run_id = prof_logger.log_run(
                            attack_string=child_payload,
                            context=context,
                            outcome=outcome,
                            source="failure_tree",
                            parent_run_id=seed["parent_run_id"],
                            perturbation=pert_meta,
                            capture_mode="first_token",
                            call_idx=None,
                            wall_time=time.time() - t_child,
                            tree_depth=tree_depth,
                            tree_root_run_id=tree_root_run_id,
                        )
                        _log_first_token_tensors(prof_logger, child_run_id, child_out)

                        children_generated    += 1
                        level_stats["n_children"] += 1
                        tree_summary["total_children"] += 1

                        if outcome["success"]:
                            tree_summary["total_successes"] += 1
                            level_stats["n_successes"] += 1
                        else:
                            tree_summary["total_failures"] += 1
                            level_stats["n_failures"] += 1
                            if level + 1 < max_levels:
                                next_failures.append({
                                    "payload":              child_payload,
                                    "token_ids":            pert["perturbed_ids"],
                                    "suite":                seed["suite"],
                                    "user_task_id":         seed["user_task_id"],
                                    "injection_task_id":    seed["injection_task_id"],
                                    "parent_run_id":        child_run_id,
                                    "tree_root_run_id":     tree_root_run_id,
                                    "vital_strings":        seed.get("vital_strings") or [],
                                    "N":                    N,
                                    "bl_info":              bl,
                                    "original_payload_ids": orig_ids,
                                })

                        logger.info("      N=%d %s utility=%s injection=%s cap=%s",
                                    N, position, outcome["utility"], outcome["success"],
                                    "ok" if child_cap_result is not None else "none")

        tree_summary["per_level"].append(level_stats)
        logger.info("Tree level %d done: %d children, %d failures, %d successes",
                    level + 1, level_stats["n_children"],
                    level_stats["n_failures"], level_stats["n_successes"])

        if level + 1 >= max_levels or not next_failures:
            if not next_failures:
                logger.info("Tree level %d: no failures collected, stopping early", level + 1)
            break

        # Decay budget and re-seed for the next level
        current_target_cell = max(1, int(current_target_cell * budget_decay))
        current_seeds = _select_diverse_failure_seeds(next_failures, n_roots, rng)
        logger.info("Tree level %d → %d: %d seeds selected (target_per_cell=%d)",
                    level + 1, level + 2, len(current_seeds), current_target_cell)

    llm.clear_prefix_cache()
    logger.info("  Failure tree done: %d children (%d failures, %d successes)",
                tree_summary["total_children"],
                tree_summary["total_failures"],
                tree_summary["total_successes"])
    return tree_summary


# ═══════════════════════════════════════════════════════════════════════
# First-token experiment loop
# ═══════════════════════════════════════════════════════════════════════

def run_first_token_experiment(
    llm: LocalHarmonyLLM,
    pipeline,
    inputs: List[dict],
    config: Optional[dict],
    prof_logger,
) -> dict:
    """Experiment loop focused on first-generated-token attention.

    For each input:
      1. Run baseline → capture prompt-only attention at first post-injection step.
      2. Gate: skip perturbation grid if baseline injection did not succeed.
      3. For each grid cell, generate distance-filtered payload perturbations,
         run each through the full pipeline, capture first-token attention.

    This is intentionally simpler than run_perturbation_experiment: no reasoning
    tokens, no multi-step profiling, just the single most interpretable signal.
    """
    cfg = _resolve_config(config)
    rng = random.Random(cfg["seed"])

    vocab_size = llm.tokenizer.vocab_size
    special_ids = set(HARMONY_SPECIAL_IDS)
    distance_fn_name = cfg["distance_fn_name"]
    feature_config = llm.feature_config

    experiment_t0 = time.time()
    failure_pool: List[dict] = []
    summary = {
        "config": cfg,
        "n_inputs": len(inputs),
        "total_baseline_runs": 0,
        "total_perturbation_runs": 0,
        "total_cells": 0,
        "total_undersampled_cells": 0,
        "total_warnings": 0,
        "input_summaries": [],
    }

    for input_idx, inp in enumerate(inputs):
        logger.info("=" * 60)
        logger.info("Input %d/%d: %s/%s/%s  payload=%d chars",
                    input_idx + 1, len(inputs),
                    inp["suite"], inp["user_task_id"], inp["injection_task_id"],
                    len(inp["payload"]))

        input_t0 = time.time()
        input_summary = {
            "input_idx": input_idx,
            "suite": inp["suite"],
            "user_task_id": inp["user_task_id"],
            "injection_task_id": inp["injection_task_id"],
            "cells": [],
        }

        original_ids = tokenize_payload(llm.tokenizer, inp["payload"])
        payload_len = len(original_ids)
        input_summary["payload_len_tokens"] = payload_len

        # Precompute original side of distance function once per payload.
        bound_dist = bind_distance_fn(
            distance_fn_name, llm.model, llm.tokenizer, llm.device, original_ids,
        )

        # Detect vital token positions that must not be perturbed.
        protected_positions: set = set()
        vital_strings = inp.get("vital_strings") or []
        if vital_strings:
            protected_positions = find_protected_positions(llm.tokenizer, original_ids, vital_strings)
            if protected_positions:
                logger.info("  Protected positions (vital tokens %s): %s",
                            vital_strings, sorted(protected_positions))

        # ── Baseline ───────────────────────────────────────────────
        logger.info("  Baseline...")
        baseline_out = profile_first_token_only(
            llm, pipeline,
            inp["suite"], inp["user_task_id"], inp["injection_task_id"],
            inp["payload"], feature_config,
            attack_payload_for_tagging=inp["payload"],
            n_reasoning_tokens=cfg["n_reasoning_tokens"],
            run_label=f"in{input_idx}_baseline",
            version=cfg["agentdojo_version"],
        )
        summary["total_warnings"] += len(baseline_out["warnings"])

        if baseline_out["task_result"] is None:
            logger.error("  Baseline pipeline failed, skipping")
            input_summary["baseline_error"] = baseline_out.get("error", "unknown")
            summary["input_summaries"].append(input_summary)
            continue

        baseline_result = baseline_out["task_result"]
        context = {
            "suite": inp["suite"],
            "user_task_id": inp["user_task_id"],
            "injection_task_id": inp["injection_task_id"],
        }
        outcome = {
            "success": baseline_result.evaluation.get("injection_success"),
            "utility": baseline_result.evaluation.get("utility"),
            "security": baseline_result.evaluation.get("security"),
        }
        baseline_run_id = prof_logger.log_run(
            attack_string=inp["payload"],
            context=context, outcome=outcome,
            source="baseline",
            perturbation={"type": "none"},
            capture_mode="first_token",
            call_idx=baseline_out.get("call_idx"),
            wall_time=baseline_out["wall_time"],
        )
        _log_first_token_tensors(prof_logger, baseline_run_id, baseline_out)
        input_summary["baseline_run_id"] = baseline_run_id
        input_summary["baseline_evaluation"] = dict(baseline_result.evaluation)
        summary["total_baseline_runs"] += 1

        logger.info("  Baseline: utility=%s  injection=%s  call_idx=%s  (%.1fs)",
                    baseline_result.evaluation.get("utility"),
                    baseline_result.evaluation.get("injection_success"),
                    baseline_out.get("call_idx"),
                    baseline_out["wall_time"])

        # ── Gate ───────────────────────────────────────────────────
        if not baseline_result.evaluation.get("injection_success"):
            logger.info("  Baseline injection failed — skipping perturbation grid")
            input_summary["skipped_reason"] = "baseline_injection_failed"
            summary["input_summaries"].append(input_summary)
            continue

        # Extract baseline token structure for direct attention capture.
        # cap["token_ids"] is the full sequence (prompt + harmony prefix +
        # first N reasoning tokens) that was fed to the model at capture time.
        bl_cap    = baseline_out.get("cap")
        bl_tagged = baseline_out.get("tagged")
        bl_token_ids   = bl_cap["token_ids"] if bl_cap is not None else None
        bl_spans       = bl_tagged.spans if bl_tagged is not None else None
        bl_rpos        = baseline_out.get("reasoning_positions") or []
        has_bl_capture = bl_token_ids is not None and bl_spans is not None and bool(bl_rpos)
        if not has_bl_capture:
            logger.warning("  Baseline capture missing — perturbation attention will not be recorded")

        # ── Prefix KV cache ────────────────────────────────────────
        # Compute and cache the KV state for the shared prompt prefix (everything
        # before the attack_payload span).  This is reused in _generate() for all
        # perturbation runs in this context, skipping 84% of prefill work each call.
        if has_bl_capture:
            payload_spans = [s for s in bl_spans if s.get("tag") == "attack_payload"]
            if payload_spans:
                payload_start = min(s["start"] for s in payload_spans)
                prefix_ids = list(bl_token_ids[:payload_start])
                t_kv = time.time()
                kv = compute_prefix_kv_cache(llm.model, prefix_ids, llm.device)
                llm.set_prefix_cache(prefix_ids, kv)
                logger.info("  Prefix KV cache computed: %d tokens, %.1fs",
                            payload_start, time.time() - t_kv)
            else:
                logger.warning("  No attack_payload span — prefix KV cache skipped")

        # ── Perturbation grid ──────────────────────────────────────
        for ptype in cfg["perturbation_types"]:
            for position in cfg["positions"]:
                for N in cfg["n_values"]:
                    if N > payload_len * cfg["max_n_fraction"] or N > payload_len:
                        continue

                    cell_label = f"{ptype}/N={N}/{position}"
                    logger.info("  Cell: %s", cell_label)

                    cell_result = generate_and_filter(
                        original_ids, ptype, N, position,
                        bound_dist, distance_fn_name,
                        tuple(cfg["distance_range"]),
                        cfg["target_per_cell"], cfg["max_candidates_per_cell"],
                        vocab_size, special_ids, rng,
                        protected_positions=protected_positions,
                    )
                    survivors = cell_result["survivors"]
                    log_cell_stats(prof_logger, baseline_run_id, cell_result["stats"])
                    summary["total_cells"] += 1
                    if cell_result["stats"]["undersampled"]:
                        summary["total_undersampled_cells"] += 1

                    cell_run_ids = []
                    cell_successes_logged = 0
                    max_succ_cell = cfg.get("max_successes_per_cell")
                    for si, pert in enumerate(survivors):
                        pert_payload = decode_payload(llm.tokenizer, pert["perturbed_ids"])
                        pert_label = f"in{input_idx}_{ptype}_N{N}_{position}_s{si}"
                        pert_t0 = time.time()

                        # ── Full pipeline run → outcome labels ──────────────
                        # Re-arm the prefix cache for this run (single-use: only the
                        # FIRST matching generate() call per run_task() uses the cache).
                        llm.rearm_prefix_cache()
                        try:
                            pert_result = run_task(
                                pipeline, llm,
                                inp["suite"], inp["user_task_id"], inp["injection_task_id"],
                                pert_payload,
                                version=cfg["agentdojo_version"],
                            )
                        except Exception as e:
                            logger.error("  %s: pipeline failed: %s", pert_label, e)
                            summary["total_warnings"] += 1
                            continue

                        t_pert_pipeline = time.time() - pert_t0

                        pert_outcome = {
                            "success": pert_result.evaluation.get("injection_success"),
                            "utility": pert_result.evaluation.get("utility"),
                            "security": pert_result.evaluation.get("security"),
                        }

                        # ── Success cap ──────────────────────────────────────
                        # Skip capture + logging for successes beyond the cap.
                        # Pipeline already ran (outcome unknown beforehand), but
                        # we avoid the attention forward pass and DB write.
                        if (max_succ_cell is not None
                                and pert_outcome["success"]
                                and cell_successes_logged >= max_succ_cell):
                            logger.debug("  %s: success cap hit (%d), skipping",
                                         pert_label, max_succ_cell)
                            continue

                        # ── Attention capture → direct token substitution ────
                        # Substitute perturbed payload tokens into the baseline
                        # token sequence and run one forward pass at the baseline
                        # reasoning positions.  No text matching needed.
                        t_cap0 = time.time()
                        pert_cap_result = None
                        if has_bl_capture:
                            pert_cap_result = capture_perturbed_attention(
                                llm.model,
                                bl_token_ids,
                                bl_spans,
                                bl_rpos,
                                pert["perturbed_ids"],
                                llm.device,
                                feature_config,
                                original_payload_ids=original_ids,
                            )
                            if pert_cap_result is None:
                                logger.debug("  %s: capture_perturbed_attention returned None", pert_label)
                                summary["total_warnings"] += 1
                        t_pert_capture = time.time() - t_cap0
                        logger.info("  %s: t_pipeline=%.1fs  t_capture=%.1fs  total=%.1fs",
                                    pert_label, t_pert_pipeline, t_pert_capture,
                                    t_pert_pipeline + t_pert_capture)

                        # Assemble pert_out dict compatible with _log_first_token_tensors.
                        # _log_first_token_tensors reads output["tagged"].spans, so we
                        # wrap the adjusted span list in a simple namespace rather than
                        # keeping the baseline TaggedConversation (whose span offsets may
                        # differ for delete perturbations).
                        if pert_cap_result is not None:
                            p_cap, p_spans, p_rpos = pert_cap_result
                            pert_out = {
                                "cap": p_cap,
                                "tagged": types.SimpleNamespace(spans=p_spans),
                                "reasoning_positions": p_rpos,
                                "call_idx": baseline_out.get("call_idx"),
                            }
                        else:
                            pert_out = {"cap": None, "tagged": None, "reasoning_positions": [], "call_idx": None}

                        pert_meta = {
                            "type": pert["perturbation_type"],
                            "N": pert["N"],
                            "position": pert["position"],
                            "position_start": pert["position_start"],
                            "position_end": pert["position_end"],
                            "original_len": pert["original_len"],
                            "perturbed_len": pert["perturbed_len"],
                        }
                        pert_run_id = prof_logger.log_run(
                            attack_string=pert_payload,
                            context=context, outcome=pert_outcome,
                            source="perturbation",
                            parent_run_id=baseline_run_id,
                            perturbation=pert_meta,
                            capture_mode="first_token",
                            call_idx=pert_out["call_idx"],
                            wall_time=time.time() - pert_t0,
                        )
                        prof_logger.log_perturbation_distances(
                            pert_run_id, baseline_run_id,
                            {pert["distance_fn_name"]: pert["distance"]},
                        )
                        _log_first_token_tensors(prof_logger, pert_run_id, pert_out)
                        cell_run_ids.append(pert_run_id)
                        summary["total_perturbation_runs"] += 1
                        if pert_outcome["success"]:
                            cell_successes_logged += 1

                        # Collect failures for the tree experiment.
                        if not pert_outcome["success"] and cfg.get("failure_tree_n_roots", 0) > 0:
                            failure_pool.append({
                                "payload": pert_payload,
                                "token_ids": pert["perturbed_ids"],
                                "suite": inp["suite"],
                                "user_task_id": inp["user_task_id"],
                                "injection_task_id": inp["injection_task_id"],
                                "parent_run_id": pert_run_id,
                                "vital_strings": inp.get("vital_strings") or [],
                                "N": pert["N"],
                                "bl_info": (
                                    {"token_ids": bl_token_ids, "spans": bl_spans, "rpos": bl_rpos}
                                    if has_bl_capture else None
                                ),
                                "original_payload_ids": list(original_ids),
                            })

                        logger.info("    s%d dist=%.4f utility=%s injection=%s cap=%s",
                                    si, pert["distance"],
                                    pert_result.evaluation.get("utility"),
                                    pert_result.evaluation.get("injection_success"),
                                    "ok" if pert_out["cap"] is not None else "none")

                    input_summary["cells"].append({
                        "cell": cell_label, "n_runs": len(cell_run_ids),
                        "run_ids": cell_run_ids,
                        "stats": cell_result["stats"],
                    })

        input_summary["wall_time"] = time.time() - input_t0
        summary["input_summaries"].append(input_summary)
        logger.info("  Input %d done in %.1fs", input_idx + 1, input_summary["wall_time"])
        llm.clear_prefix_cache()

    # ── Failure tree ───────────────────────────────────────────────────
    if cfg.get("failure_tree_n_roots", 0) > 0:
        if failure_pool:
            seeds = _select_diverse_failure_seeds(
                failure_pool, cfg["failure_tree_n_roots"], rng,
            )
            logger.info("Failure tree: %d seeds selected from %d level-1 failures",
                        len(seeds), len(failure_pool))
            tree_summary = run_failure_tree_experiment(llm, pipeline, seeds, cfg, prof_logger)
            summary["failure_tree"] = tree_summary
        else:
            logger.info("Failure tree: no level-1 failures collected, skipping")

    summary["total_wall_time"] = time.time() - experiment_t0
    prof_logger.logger.log(
        summary, variable_name="experiment_summary",
        event="first_token_experiment_summary",
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════════════════════════════

def run_perturbation_experiment(
    llm: LocalHarmonyLLM,
    pipeline,
    inputs: List[dict],
    config: Optional[dict],
    prof_logger,
) -> dict:
    """Run the full perturbation experiment.

    Args:
        llm:          LocalHarmonyLLM (has model, tokenizer, device)
        pipeline:     AgentPipeline
        inputs:       List of dicts, each with:
                          suite, user_task_id, injection_task_id, payload
        config:       Experiment configuration (merged over DEFAULT_CONFIG)
        prof_logger:  ProfilingLogger instance

    Returns:
        Summary dict with per-input and per-cell stats.
    """
    cfg = _resolve_config(config)
    rng = random.Random(cfg["seed"])

    vocab_size = llm.tokenizer.vocab_size
    special_ids = set(HARMONY_SPECIAL_IDS)
    distance_fn_name = cfg["distance_fn_name"]
    feature_config = llm.feature_config

    experiment_t0 = time.time()
    experiment_summary = {
        "config": cfg,
        "n_inputs": len(inputs),
        "input_summaries": [],
        "total_baseline_runs": 0,
        "total_perturbation_runs": 0,
        "total_cells": 0,
        "total_undersampled_cells": 0,
        "total_warnings": 0,
    }

    for input_idx, inp in enumerate(inputs):
        logger.info("=" * 60)
        logger.info("Input %d/%d: %s / %s / %s",
                     input_idx + 1, len(inputs),
                     inp["suite"], inp["user_task_id"], inp["injection_task_id"])
        logger.info("  Payload: %s...", inp["payload"][:80])

        input_t0 = time.time()
        input_summary = {
            "input_idx": input_idx,
            "suite": inp["suite"],
            "user_task_id": inp["user_task_id"],
            "injection_task_id": inp["injection_task_id"],
            "payload_len_chars": len(inp["payload"]),
            "cells": [],
        }

        # ── Tokenize payload ───────────────────────────────────────
        original_ids = tokenize_payload(llm.tokenizer, inp["payload"])
        payload_len = len(original_ids)
        input_summary["payload_len_tokens"] = payload_len
        logger.info("  Payload: %d tokens", payload_len)

        # Precompute original side of distance function once per payload.
        bound_dist = bind_distance_fn(
            distance_fn_name, llm.model, llm.tokenizer, llm.device, original_ids,
        )

        # Detect vital token positions that must not be perturbed.
        protected_positions: set = set()
        vital_strings = inp.get("vital_strings") or []
        if vital_strings:
            protected_positions = find_protected_positions(llm.tokenizer, original_ids, vital_strings)
            if protected_positions:
                logger.info("  Protected positions (vital tokens %s): %s",
                            vital_strings, sorted(protected_positions))

        # ── Run baseline ───────────────────────────────────────────
        logger.info("  Running baseline...")
        baseline_output = profile_single_run(
            llm, pipeline,
            inp["suite"], inp["user_task_id"], inp["injection_task_id"],
            inp["payload"],
            feature_config,
            attack_payload_for_tagging=inp["payload"],
            profile_from_injection=cfg["profile_from_injection"],
            profile_fine_groups=cfg["profile_fine_groups"],
            save_tensors=cfg["save_tensors"],
            run_label=f"input{input_idx}_baseline",
            version=cfg["agentdojo_version"],
        )

        if baseline_output["task_result"] is None:
            logger.error("  Baseline pipeline failed, skipping input")
            input_summary["baseline_error"] = baseline_output.get("error", "unknown")
            experiment_summary["input_summaries"].append(input_summary)
            continue

        baseline_result = baseline_output["task_result"]
        baseline_run_id = log_baseline_run(
            prof_logger, baseline_result, baseline_output, inp, cfg,
        )
        input_summary["baseline_run_id"] = baseline_run_id
        input_summary["baseline_evaluation"] = dict(baseline_result.evaluation)
        input_summary["baseline_wall_time"] = baseline_output["wall_time"]
        input_summary["baseline_warnings"] = baseline_output["warnings"]
        experiment_summary["total_baseline_runs"] += 1

        logger.info("  Baseline: utility=%s, injection_success=%s (%.1fs)",
                     baseline_result.evaluation.get("utility"),
                     baseline_result.evaluation.get("injection_success"),
                     baseline_output["wall_time"])

        # ── Gate: only profile perturbations if baseline succeeded ──
        baseline_succeeded = baseline_result.evaluation.get("injection_success") is True
        if not baseline_succeeded:
            logger.info("  Baseline injection did NOT succeed (security=%s) — "
                        "skipping perturbation grid for this input",
                        baseline_result.evaluation.get("security"))
            input_summary["skipped_reason"] = "baseline_injection_failed"
            experiment_summary["input_summaries"].append(input_summary)
            continue

        # ── Sweep grid ─────────────────────────────────────────────
        for ptype in cfg["perturbation_types"]:
            for position in cfg["positions"]:
                for N in cfg["n_values"]:

                    # Skip cells where N is too large
                    if N > payload_len * cfg["max_n_fraction"]:
                        logger.info("  Skipping %s/N=%d/%s (N > %.0f%% of %d)",
                                     ptype, N, position,
                                     cfg["max_n_fraction"] * 100, payload_len)
                        continue

                    if N > payload_len:
                        logger.info("  Skipping %s/N=%d/%s (N > payload_len=%d)",
                                     ptype, N, position, payload_len)
                        continue

                    cell_label = f"{ptype}/N={N}/{position}"
                    logger.info("  Cell: %s", cell_label)

                    # ── Generate and filter ─────────────────────────
                    cell_result = generate_and_filter(
                        original_ids,
                        ptype, N, position,
                        bound_dist, distance_fn_name,
                        tuple(cfg["distance_range"]),
                        cfg["target_per_cell"],
                        cfg["max_candidates_per_cell"],
                        vocab_size, special_ids, rng,
                        protected_positions=protected_positions,
                    )

                    cell_stats = cell_result["stats"]
                    survivors = cell_result["survivors"]

                    log_cell_stats(prof_logger, baseline_run_id, cell_stats)
                    experiment_summary["total_cells"] += 1
                    if cell_stats["undersampled"]:
                        experiment_summary["total_undersampled_cells"] += 1

                    logger.info("    Generated %d, in range %d/%d",
                                 cell_stats["generated"],
                                 cell_stats["in_range"],
                                 cell_stats["target"])

                    # ── Run each survivor ───────────────────────────
                    cell_run_ids = []
                    for si, pert in enumerate(survivors):
                        pert_payload = decode_payload(
                            llm.tokenizer, pert["perturbed_ids"])

                        pert_label = (f"input{input_idx}_{ptype}_"
                                      f"N{N}_{position}_s{si}")

                        logger.info("    Survivor %d/%d (dist=%.4f): running...",
                                     si + 1, len(survivors), pert["distance"])

                        pert_output = profile_single_run(
                            llm, pipeline,
                            inp["suite"], inp["user_task_id"],
                            inp["injection_task_id"],
                            pert_payload,
                            feature_config,
                            attack_payload_for_tagging=pert_payload,
                            profile_from_injection=cfg["profile_from_injection"],
                            profile_fine_groups=cfg["profile_fine_groups"],
                            save_tensors=cfg["save_tensors"],
                            run_label=pert_label,
                            version=cfg["agentdojo_version"],
                        )

                        if pert_output["task_result"] is None:
                            logger.warning("    Survivor %d: pipeline failed",
                                           si + 1)
                            experiment_summary["total_warnings"] += 1
                            continue

                        pert_result = pert_output["task_result"]
                        pert_run_id = log_perturbation_run(
                            prof_logger,
                            pert_result, pert_output,
                            inp, cfg,
                            baseline_run_id, pert, pert_payload,
                        )
                        cell_run_ids.append(pert_run_id)
                        experiment_summary["total_perturbation_runs"] += 1
                        experiment_summary["total_warnings"] += len(
                            pert_output["warnings"])

                        logger.info(
                            "      utility=%s injection=%s (%.1fs, %d warnings)",
                            pert_result.evaluation.get("utility"),
                            pert_result.evaluation.get("injection_success"),
                            pert_output["wall_time"],
                            len(pert_output["warnings"]),
                        )

                    cell_summary = {
                        "cell": cell_label,
                        "perturbation_type": ptype,
                        "N": N,
                        "position": position,
                        "stats": cell_stats,
                        "n_runs": len(cell_run_ids),
                        "run_ids": cell_run_ids,
                    }
                    input_summary["cells"].append(cell_summary)

        input_summary["wall_time"] = time.time() - input_t0
        experiment_summary["input_summaries"].append(input_summary)

        logger.info("  Input %d done in %.1fs",
                     input_idx + 1, input_summary["wall_time"])

    experiment_summary["total_wall_time"] = time.time() - experiment_t0

    logger.info("=" * 60)
    logger.info("Experiment complete: %d baselines, %d perturbations, "
                "%d cells (%d undersampled), %.1fs total",
                experiment_summary["total_baseline_runs"],
                experiment_summary["total_perturbation_runs"],
                experiment_summary["total_cells"],
                experiment_summary["total_undersampled_cells"],
                experiment_summary["total_wall_time"])

    # Log experiment summary
    prof_logger.logger.log(
        experiment_summary,
        variable_name="experiment_summary",
        event="perturbation_experiment_summary",
    )

    return experiment_summary