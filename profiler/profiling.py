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
from conversation_tagger import TaggedConversation, tag_conversation, tag_gen_tokens, TAGS

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

def reconstruct_step_tokens(task_result, call_idx: int) -> List[int]:
    """Get prompt + raw gen (including reasoning) for one inference step."""
    if call_idx >= len(task_result.call_log):
        raise IndexError(
            f"call_idx={call_idx} but only {len(task_result.call_log)} calls logged"
        )
    entry = task_result.call_log[call_idx]
    return entry["prompt_token_ids"] + entry["gen_token_ids"]


def tag_step_with_reasoning(
    formatter,
    task_result,
    call_idx: int,
    *,
    attack_payload: Optional[str] = None,
    attack_prefix: Optional[str] = None,
    attack_suffix: Optional[str] = None,
) -> Tuple[List[int], TaggedConversation]:
    """Tag a full inference step INCLUDING reasoning tokens.

    Combines:
    1. Tag the prompt part (stripped conversation) via tag_conversation
    2. Tag the generation part (reasoning + tool call) via tag_gen_tokens

    Returns (full_token_ids, tagged_conversation).
    """
    from inference_runner import chat_messages_to_dicts

    entry = task_result.call_log[call_idx]
    prompt_ids = entry["prompt_token_ids"]
    gen_ids = entry["gen_token_ids"]
    full_ids = prompt_ids + gen_ids

    # Part 1: Tag the prompt
    messages_in = entry["messages_in"]
    prompt_messages = list(task_result.messages[:messages_in])

    simple_msgs = chat_messages_to_dicts(prompt_messages)
    tool_descs = agentdojo_tools_to_descriptions(task_result.runtime)

    prompt_tagged = tag_conversation(
        formatter, simple_msgs, tool_descs,
        attack_payload=attack_payload,
        attack_prefix=attack_prefix,
        attack_suffix=attack_suffix,
    )

    # Part 2: Tag the generation
    prompt_step = max((s["step"] for s in prompt_tagged.spans), default=0)
    gen_step = prompt_step

    gen_tags, gen_spans = tag_gen_tokens(
        formatter, gen_ids,
        offset=len(prompt_ids),
        step=gen_step,
    )

    # Combine
    all_spans = list(prompt_tagged.spans) + gen_spans
    full_len = len(full_ids)

    positions = {t: [] for t in TAGS}
    for tag in TAGS:
        if tag in prompt_tagged.masks:
            idxs = prompt_tagged.masks[tag].nonzero(as_tuple=True)[0].tolist()
            positions[tag].extend(idxs)
    for i, t in enumerate(gen_tags):
        if t is not None and t in positions:
            positions[t].append(len(prompt_ids) + i)

    masks = {}
    for tag in TAGS:
        m = torch.zeros(full_len, dtype=torch.bool)
        if positions[tag]:
            m[torch.tensor(positions[tag], dtype=torch.long)] = True
        masks[tag] = m

    return full_ids, TaggedConversation(tokens=full_ids, masks=masks, spans=all_spans)


def capture_reasoning_entry_attention(
    model,
    formatter,
    task_result,
    device: str,
    feature_config: FeatureConfig,
    *,
    attack_payload: Optional[str] = None,
    attack_prefix: Optional[str] = None,
    attack_suffix: Optional[str] = None,
    n_reasoning_tokens: int = 10,
) -> Optional[Tuple[dict, "TaggedConversation", int, List[int]]]:
    """Capture attention at the first N reasoning content tokens after injection.

    The model's generation always begins with a fixed harmony control prefix:
        <|channel|> analysis <|message|>
    which is deterministic and identical across all runs for a given task.
    The first *content* tokens — the actual reasoning — start immediately after
    that prefix.

    Strategy:
      1. Take prompt_ids from the first post-injection call_log entry.
      2. Scan the corresponding gen_ids for the first TOKEN_MESSAGE, which marks
         the end of the control prefix.
      3. Build sequence = prompt_ids + gen_ids[:message_pos + 1 + n_reasoning_tokens].
         This includes the fixed prefix + the first n reasoning tokens.
      4. Capture with attention_last_row_only=False and capture_positions pointing
         at those n reasoning token positions.

    Because the control prefix length is fixed, all captured positions are at the
    same offset relative to the prompt end across all perturbation runs, making
    the attention maps directly comparable.

    Returns (capture_dict, tagged_prompt, call_idx, reasoning_positions) or None.
    reasoning_positions: list of absolute token indices in the full sequence that
                         were captured (useful for interpretation).
    """
    from harmony_format import TOKEN_MESSAGE

    post_inj = get_all_call_indices_after_injection(
        task_result, attack_payload=attack_payload
    )
    if not post_inj:
        return None

    call_idx = post_inj[0]
    entry = task_result.call_log[call_idx]
    prompt_ids = entry["prompt_token_ids"]
    gen_ids = entry["gen_token_ids"]

    # Find where reasoning content starts: first TOKEN_MESSAGE in gen_ids
    message_pos = None
    for i, tid in enumerate(gen_ids):
        if tid == TOKEN_MESSAGE:
            message_pos = i
            break

    if message_pos is None:
        # No analysis channel found — fall back to position 0
        message_pos = -1

    reasoning_start = message_pos + 1  # first reasoning content token in gen_ids

    # How many reasoning tokens are actually available?
    available = len(gen_ids) - reasoning_start
    n_actual = min(n_reasoning_tokens, available)

    if n_actual <= 0:
        return None

    # Build the sequence: prompt + fixed control prefix + first n reasoning tokens
    prefix_end = reasoning_start + n_actual          # exclusive, in gen_ids coords
    full_ids = prompt_ids + gen_ids[:prefix_end]
    prompt_len = len(prompt_ids)

    # Absolute positions of the n reasoning tokens in full_ids
    reasoning_positions = [prompt_len + reasoning_start + i for i in range(n_actual)]

    # Tag the prompt portion (for span groups)
    messages_in = entry["messages_in"]
    from inference_runner import chat_messages_to_dicts
    simple_msgs = chat_messages_to_dicts(list(task_result.messages[:messages_in]))
    tool_descs = agentdojo_tools_to_descriptions(task_result.runtime)
    tagged = tag_conversation(
        formatter, simple_msgs, tool_descs,
        attack_payload=attack_payload,
        attack_prefix=attack_prefix,
        attack_suffix=attack_suffix,
    )

    # Capture: full matrix at exactly the reasoning positions.
    # Propagate all capture flags from the caller's config.
    reasoning_config = FeatureConfig(
        layers=feature_config.layers,
        capture_attention=True,
        capture_hidden_states=feature_config.capture_hidden_states,
        capture_logit_lens=feature_config.capture_logit_lens,
        capture_value_aggregates=feature_config.capture_value_aggregates,
        capture_mlp_output=feature_config.capture_mlp_output,
        attention_last_row_only=False,
        capture_positions=reasoning_positions,
        storage_device=feature_config.storage_device,
        logit_lens_top_k=feature_config.logit_lens_top_k,
    )
    cap = capture_forward_pass(model, full_ids, device, reasoning_config)
    return cap, tagged, call_idx, reasoning_positions


def _normalize_ws(text: str) -> str:
    """Collapse literal escape sequences and whitespace runs to single spaces.

    Mirrors the normalization in HarmonyFormatter.find_injection_token_span so
    that payload detection is consistent regardless of YAML folding, line-wrap,
    or JSON serialization that the environment data goes through.
    """
    import re
    text = re.sub(r'\\[ntr]\s*', ' ', text)   # literal \n \t \r → space
    text = re.sub(r'\s+', ' ', text)            # any whitespace run → space
    return text.strip()


def get_all_call_indices_after_injection(task_result, attack_payload=None):
    """Find all call_log indices where injection is visible in the prompt.

    Finds the first tool message in task_result.messages whose text contains
    attack_payload (matched after whitespace normalization to survive YAML
    folding/line-wrapping), then returns every call_log entry whose
    messages_in count is past that message index.
    """
    if not attack_payload:
        return list(range(len(task_result.call_log)))

    norm_payload = _normalize_ws(attack_payload)

    injection_msg_idx = None
    for msg_idx, msg in enumerate(task_result.messages):
        if msg["role"] == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("content", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            if norm_payload in _normalize_ws(content):
                injection_msg_idx = msg_idx
                break

    if injection_msg_idx is None:
        # Text match failed (e.g. YAML round-trip transformed the payload).
        # Fall back: find the first tool message in the conversation and treat
        # everything after it as post-injection.  This is correct for the common
        # AgentDojo pattern where the injection arrives in the first tool response.
        for msg_idx, msg in enumerate(task_result.messages):
            if msg["role"] == "tool":
                injection_msg_idx = msg_idx
                break

    if injection_msg_idx is None:
        return []

    return [
        call_idx
        for call_idx, entry in enumerate(task_result.call_log)
        if entry["messages_in"] > injection_msg_idx
    ]


def _find_subseq(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return the start index of the first occurrence of needle in haystack, or None."""
    n = len(needle)
    if n == 0:
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return None


def capture_perturbed_attention(
    model,
    base_token_ids: List[int],
    base_spans: list,
    base_reasoning_positions: List[int],
    perturbed_payload_ids: List[int],
    device: str,
    feature_config: "FeatureConfig",
    original_payload_ids: Optional[List[int]] = None,
) -> Optional[Tuple[dict, list, List[int]]]:
    """Capture attention for a perturbed payload using the baseline token structure.

    Replaces the attack_payload region in base_token_ids with perturbed_payload_ids,
    then runs a single forward pass at the (adjusted) reasoning positions.

    Works for any perturbation type:
    - flip   (same length): span boundaries are identical.
    - delete (shorter):     spans and reasoning positions after the payload are
                            shifted left by the length difference.

    The baseline's TaggedConversation spans are re-used and adjusted in-place,
    so no re-tagging or pipeline re-run is needed.

    Args:
        model:                    The language model.
        base_token_ids:           Full token id list from the baseline capture
                                  (cap["token_ids"]).
        base_spans:               Span dicts from the baseline's TaggedConversation.
        base_reasoning_positions: Absolute token indices captured in the baseline
                                  (baseline_out["reasoning_positions"]).
        perturbed_payload_ids:    New payload token ids (any length).
        device:                   Torch device string.
        feature_config:           Feature capture config.
        original_payload_ids:     Token ids of the original (unperturbed) payload.
                                  When provided, used as a fallback to locate the
                                  payload in base_token_ids if no attack_payload span
                                  is present (handles cases where text-based tagging
                                  failed, e.g. due to YAML round-trip transforms).

    Returns:
        (cap, adjusted_spans, adjusted_reasoning_positions) or None if no
        attack_payload span is found.
    """
    payload_spans = [s for s in base_spans if s.get("tag") == "attack_payload"]

    if not payload_spans and original_payload_ids:
        # Text-based tagging failed (e.g. YAML transformed the payload text).
        # Fall back: locate original_payload_ids as an exact token subsequence.
        start = _find_subseq(list(base_token_ids), list(original_payload_ids))
        if start is not None:
            end = start + len(original_payload_ids)
            logger.debug(
                "capture_perturbed_attention: payload located by token subsequence "
                "match at [%d, %d) (text-based tagging had failed)", start, end
            )
            # Synthesise an attack_payload span so the rest of the logic is uniform.
            synthetic_span = {
                "tag": "attack_payload",
                "start": start, "end": end,
                "msg_idx": -1, "step": 0, "role": "tool", "channel": None,
            }
            base_spans = list(base_spans) + [synthetic_span]
            payload_spans = [synthetic_span]

    if not payload_spans:
        logger.warning("capture_perturbed_attention: no attack_payload span in base_spans")
        return None

    payload_start = min(s["start"] for s in payload_spans)
    payload_end   = max(s["end"]   for s in payload_spans)
    len_delta     = len(perturbed_payload_ids) - (payload_end - payload_start)

    full_ids = (
        list(base_token_ids[:payload_start])
        + list(perturbed_payload_ids)
        + list(base_token_ids[payload_end:])
    )

    # Adjust spans: everything after the payload shifts by len_delta.
    # The attack_payload span(s) themselves shrink/grow at their end.
    adjusted_spans = []
    for s in base_spans:
        ns = dict(s)
        if s["start"] >= payload_end:
            ns["start"] = s["start"] + len_delta
            ns["end"]   = s["end"]   + len_delta
        elif s["end"] > payload_start:
            # overlaps the payload region — clamp end to new payload boundary
            ns["end"] = min(s["end"], payload_end) + len_delta
        adjusted_spans.append(ns)

    # Adjust reasoning positions: shift anything that was after the payload end.
    adjusted_reasoning_positions = [
        p + len_delta if p >= payload_end else p
        for p in base_reasoning_positions
    ]

    reasoning_config = FeatureConfig(
        layers=feature_config.layers,
        capture_attention=True,
        capture_hidden_states=feature_config.capture_hidden_states,
        capture_logit_lens=feature_config.capture_logit_lens,
        capture_value_aggregates=feature_config.capture_value_aggregates,
        capture_mlp_output=feature_config.capture_mlp_output,
        attention_last_row_only=False,
        capture_positions=adjusted_reasoning_positions,
        storage_device=feature_config.storage_device,
        logit_lens_top_k=feature_config.logit_lens_top_k,
    )
    cap = capture_forward_pass(model, full_ids, device, reasoning_config)
    return cap, adjusted_spans, adjusted_reasoning_positions


def capture_step_with_reasoning(
    model,
    formatter,
    task_result,
    call_idx: int,
    device: str,
    feature_config: FeatureConfig,
    *,
    attack_payload: Optional[str] = None,
    attack_prefix: Optional[str] = None,
    attack_suffix: Optional[str] = None,
) -> Tuple[dict, TaggedConversation]:
    """Tag + capture a step with reasoning tokens included.

    Convenience wrapper. Returns (capture_dict, tagged).
    """
    full_ids, tagged = tag_step_with_reasoning(
        formatter, task_result, call_idx,
        attack_payload=attack_payload,
        attack_prefix=attack_prefix,
        attack_suffix=attack_suffix,
    )
    cap = capture_forward_pass(model, full_ids, device, feature_config)
    return cap, tagged


# ── Capture (GPU) ──────────────────────────────────────────────────────

def _get_lm_head(model):
    """Return (lm_head, final_layer_norm) for logit lens computation, or (None, None)."""
    lm_head = getattr(model, "lm_head", None)
    norm = None
    try:
        norm = model.model.norm
    except AttributeError:
        pass
    return lm_head, norm


def capture_forward_pass(
    model,
    token_ids: List[int],
    device: str,
    feature_config: FeatureConfig,
) -> dict:
    """Run a forward pass and return raw tensors as a plain dict.

    Returns a dict with all captured signals:
        "attention"        — {layer_idx: tensor}  attention weights
        "sink_mass"        — {layer_idx: tensor}  probability mass to learned sink scalar
        "hidden_states"    — {layer_idx: tensor}  residual stream at layer output
        "value_aggregates" — {layer_idx: tensor}  sum_i(alpha_i v_i) before o_proj
        "mlp_outputs"      — {layer_idx: tensor}  MLP/MoE additive contribution
        "logit_lens"       — {layer_idx: dict}    top-k logit lens projections (if enabled)
        "token_ids"        — list[int]
        "seq_len"          — int
    Keys with no data (disabled in config or hook not found) hold empty dicts.
    """
    import time as _time
    capture = FeatureCapture(model, feature_config)
    input_tensor = torch.tensor([token_ids], device=device)
    model.eval()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _t0 = _time.time()
    with torch.no_grad(), capture.active() as features:
        model(input_ids=input_tensor, output_attentions=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    logger.debug("capture_forward_pass: seq_len=%d  gpu_time=%.2fs", len(token_ids), _time.time() - _t0)

    result = {
        "attention":        dict(features.attention),
        "sink_mass":        dict(features.sink_mass),
        "hidden_states":    dict(features.hidden_states),
        "value_aggregates": dict(features.value_aggregates),
        "mlp_outputs":      dict(features.mlp_outputs),
        "logit_lens":       {},
        "token_ids":        list(token_ids),
        "seq_len":          len(token_ids),
    }

    if feature_config.capture_logit_lens and features.hidden_states:
        lm_head, norm = _get_lm_head(model)
        if lm_head is not None:
            result["logit_lens"] = capture.compute_logit_lens(
                lm_head, norm, top_k=feature_config.logit_lens_top_k
            )
        else:
            logger.warning("capture_logit_lens=True but model has no lm_head attribute")

    return result


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