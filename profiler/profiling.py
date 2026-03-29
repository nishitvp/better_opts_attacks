"""
profiling.py
============

Profiling functions for GPT-OSS prompt-injection analysis.

Three concerns, all plain functions:
    1. capture_forward_pass  — run model, return raw tensors as a dict (GPU)
    2. tag_task_result       — tag a TaskResult's messages (CPU)
    3. compute_*             — analyze stored tensors with span groups (CPU)

Design: run the expensive GPU work once, get back a dict, then analyze
with different span groups as many times as you want.

Usage:
    from profiling import capture_forward_pass, compute_attention_mass, \
                          compute_scalar_features, tag_task_result, save_capture, load_capture

    result = run_task(pipeline, llm, ...)

    # Tag (CPU)
    tagged = tag_task_result(llm.formatter, result, attack_prefix="<IMPORTANT>", ...)

    # Capture at a specific step (GPU)
    trunc = tagged.at_step(2)
    cap = capture_forward_pass(llm.model, trunc.tokens, llm.device, llm.feature_config)

    # Analyze with different groupings (CPU, cheap, do many times)
    coarse = trunc.to_span_groups(fine=False)
    fine   = trunc.to_span_groups(fine=True)
    mass   = compute_attention_mass(cap, coarse, trunc.seq_len)
    feats  = compute_scalar_features(cap, fine, trunc.seq_len)

    # Save/load
    save_capture(cap, "captures/step_2.pt")
    cap2 = load_capture("captures/step_2.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import ChatMessage

from feature_hooks import FeatureCapture, FeatureConfig
from harmony_format import agentdojo_tools_to_descriptions
from conversation_tagger import TaggedConversation, tag_conversation

logger = logging.getLogger(__name__)


# ── Tagging (CPU) ───────────────────────────────────────────────────────

def tag_task_result(
    formatter,
    task_result,
    *,
    attack_payload: Optional[str] = None,
    attack_prefix: Optional[str] = None,
    attack_suffix: Optional[str] = None,
) -> TaggedConversation:
    """Tag a TaskResult's messages.

    Convenience wrapper: pulls messages + runtime from a TaskResult,
    converts to simple dicts, and calls tag_conversation.
    """
    from inference_runner import chat_messages_to_dicts
    simple_msgs = chat_messages_to_dicts(task_result.messages)
    tool_descs = agentdojo_tools_to_descriptions(task_result.runtime)
    return tag_conversation(
        formatter, simple_msgs, tool_descs,
        attack_payload=attack_payload,
        attack_prefix=attack_prefix,
        attack_suffix=attack_suffix,
    )


# ── Capture (GPU) ──────────────────────────────────────────────────────

def capture_forward_pass(
    model,
    token_ids: List[int],
    device: str,
    feature_config: FeatureConfig,
) -> dict:
    """Run a forward pass and return raw tensors as a plain dict.

    Returns:
        {
            "attention":     {layer_idx: tensor},   # on CPU
            "hidden_states": {layer_idx: tensor},   # on CPU
            "token_ids":     list[int],
            "seq_len":       int,
        }
    """
    capture = FeatureCapture(model, feature_config)
    input_tensor = torch.tensor([token_ids], device=device)
    model.eval()

    with torch.no_grad(), capture.active() as features:
        output = model(input_ids=input_tensor, output_attentions=True)

    return {
        "attention":     {k: v.cpu().clone() for k, v in features.attention.items()},
        "hidden_states": {k: v.cpu().clone() for k, v in features.hidden_states.items()},
        "token_ids":     list(token_ids),
        "seq_len":       len(token_ids),
    }


def capture_at_step(
    model,
    tagged: TaggedConversation,
    step: int,
    device: str,
    feature_config: FeatureConfig,
) -> Tuple[dict, TaggedConversation]:
    """Truncate tagged conversation to step, then capture.

    Returns (capture_dict, truncated_tagged).
    """
    trunc = tagged.at_step(step)
    cap = capture_forward_pass(model, trunc.tokens, device, feature_config)
    return cap, trunc


# ── Analysis (CPU) ──────────────────────────────────────────────────────

def compute_attention_mass(
    capture: dict,
    span_groups: Dict[str, List[Tuple[int, int]]],
    seq_len: Optional[int] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute attention mass on each span group at each layer.

    For each (layer, head), sums attention weights falling within
    each span group's token ranges.

    Args:
        capture:     Dict from capture_forward_pass().
        span_groups: {group_name: [(start, end), ...]}.
        seq_len:     Override (default: from capture).

    Returns:
        {group_name: {layer_idx: tensor(num_heads)}}
    """
    attention = capture["attention"]
    sl = seq_len if seq_len is not None else capture["seq_len"]

    if not attention:
        return {}

    masks: Dict[str, torch.Tensor] = {}
    for name, sp in span_groups.items():
        mask = torch.zeros(sl, dtype=torch.bool)
        for start, end in sp:
            mask[start:min(end, sl)] = True
        masks[name] = mask

    result: Dict[str, Dict[int, torch.Tensor]] = {name: {} for name in span_groups}

    for layer_idx, attn in attention.items():
        for name, mask in masks.items():
            m = mask.to(attn.device)

            if attn.dim() == 2:
                # Last-row-only: (num_heads, seq_len)
                actual_seq = attn.shape[-1]
                if actual_seq != sl:
                    m = m[:actual_seq] if actual_seq < sl else torch.nn.functional.pad(m, (0, actual_seq - sl))
                mass = (attn * m.unsqueeze(0)).sum(dim=-1)
            elif attn.dim() == 3:
                # Full: (num_heads, q_len, k_len)
                actual_klen = attn.shape[-1]
                if actual_klen != sl:
                    m = m[:actual_klen] if actual_klen < sl else torch.nn.functional.pad(m, (0, actual_klen - sl))
                mass = (attn * m.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean(dim=-1)
            else:
                logger.warning("Unexpected attention shape at layer %d: %s", layer_idx, attn.shape)
                continue

            result[name][layer_idx] = mass.cpu()

    return result


def compute_scalar_features(
    capture: dict,
    span_groups: Dict[str, List[Tuple[int, int]]],
    seq_len: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract per-(layer, head) scalar features from a capture.

    Returns flat list of dicts, each with:
        layer, head, feature_type,
        {group_name}_attention_mass for each group,
        residual_norm (at head==0 only).
    """
    attention = capture["attention"]
    hidden_states = capture["hidden_states"]
    sl = seq_len if seq_len is not None else capture["seq_len"]

    attn_mass = compute_attention_mass(capture, span_groups, sl)
    features = []

    captured_layers = sorted(set(attention.keys()) | set(hidden_states.keys()))

    for layer_idx in captured_layers:
        num_heads = 0
        if layer_idx in attention:
            num_heads = attention[layer_idx].shape[0]

        if num_heads == 0:
            entry = {"layer": layer_idx, "head": None, "feature_type": "residual"}
            if layer_idx in hidden_states:
                entry["residual_norm"] = hidden_states[layer_idx][-1].norm().item()
            features.append(entry)
            continue

        for head_idx in range(num_heads):
            entry = {"layer": layer_idx, "head": head_idx, "feature_type": "attention"}

            for group_name in span_groups:
                if group_name in attn_mass and layer_idx in attn_mass[group_name]:
                    mass_tensor = attn_mass[group_name][layer_idx]
                    entry[f"{group_name}_attention_mass"] = mass_tensor[head_idx].item()

            if head_idx == 0 and layer_idx in hidden_states:
                entry["residual_norm"] = hidden_states[layer_idx][-1].norm().item()

            features.append(entry)

    return features


# ── Persistence ─────────────────────────────────────────────────────────

def save_capture(capture: dict, path: str):
    """Save a capture dict to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(capture, path)
    logger.info("Saved capture to %s", path)


def load_capture(path: str) -> dict:
    """Load a capture dict from disk."""
    cap = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Loaded capture from %s", path)
    return cap