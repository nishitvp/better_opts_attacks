"""
conversation_tagger.py
======================

Token-level semantic tagger for GPT-OSS harmony-format conversations.

Tags every token with exactly one label from a fixed taxonomy.  Output
feeds directly into profiling.compute_attention_mass() via span groups.

Granular frame tags let you measure attention mass on specific special
tokens (message boundaries vs channel markers vs constrain tokens etc.).
Coarse mode collapses them all to "control" for backward compat.

Usage:
    tagged = tag_conversation(formatter, messages, tool_descs,
                              attack_payload="Ignore previous...",
                              attack_prefix="<IMPORTANT>",
                              attack_suffix="</IMPORTANT>")
    groups = tagged.to_span_groups()              # coarse
    groups = tagged.to_span_groups(fine=True)      # per-tag
    truncated = tagged.at_step(1)                  # for step-level profiling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from harmony_format import (
    HarmonyFormatter,
    TOKEN_START,
    TOKEN_END,
    TOKEN_MESSAGE,
    TOKEN_CHANNEL,
    TOKEN_CONSTRAIN,
    TOKEN_RETURN,
    TOKEN_CALL,
    TERMINAL_TOKENS,
)

# ── Taxonomy ────────────────────────────────────────────────────────────

# Granular frame (control) tags
FRAME_TAGS = [
    "frame_boundary",        # <|start|>, <|end|>, <|return|>, <|call|>
    "frame_message",         # <|message|>
    "frame_channel",         # <|channel|> special token
    "frame_constrain",       # <|constrain|> special token
    "frame_role",            # role/recipient text after <|start|>  (e.g. "assistant", "developer")
    "frame_channel_name",    # channel name text after <|channel|>  (e.g. "commentary", "final")
    "frame_constrain_type",  # constrain type text after <|constrain|> (e.g. " json")
    "frame_metadata",        # other text in framing regions (safety net)
]

# Content tags
CONTENT_TAGS = [
    "system_meta",
    "developer_instructions",
    "developer_tools",
    "user_instruction",
    "tool_env_data",
    "attack_payload",
    "attack_prefix",
    "attack_suffix",
    "assistant_reasoning",
    "assistant_tool_call",
    "assistant_commentary",
    "assistant_final",
]

TAGS = FRAME_TAGS + CONTENT_TAGS

# Special token ID → granular frame tag
_SPECIAL_TO_TAG = {
    TOKEN_START:     "frame_boundary",
    TOKEN_END:       "frame_boundary",
    TOKEN_RETURN:    "frame_boundary",
    TOKEN_CALL:      "frame_boundary",
    TOKEN_MESSAGE:   "frame_message",
    TOKEN_CHANNEL:   "frame_channel",
    TOKEN_CONSTRAIN: "frame_constrain",
}

# Coarse grouping (fine tag → coarse group)
COARSE = {
    "frame_boundary":        "control",
    "frame_message":         "control",
    "frame_channel":         "control",
    "frame_constrain":       "control",
    "frame_role":            "control",
    "frame_channel_name":    "control",
    "frame_constrain_type":  "control",
    "frame_metadata":        "control",
    "system_meta":            "system",
    "developer_instructions": "system",
    "developer_tools":        "system",
    "user_instruction":       "user",
    "tool_env_data":          "tool_clean",
    "attack_payload":         "payload",
    "attack_prefix":          "tool_injected",
    "attack_suffix":          "tool_injected",
    "assistant_reasoning":    "assistant",
    "assistant_tool_call":    "assistant",
    "assistant_commentary":   "assistant",
    "assistant_final":        "assistant",
}


# ── Output ──────────────────────────────────────────────────────────────

@dataclass
class TaggedConversation:
    """Just a container for tagging results — tokens, masks, spans."""

    tokens: List[int]
    masks: Dict[str, torch.BoolTensor]       # tag → bool[seq_len]
    spans: List[dict]                         # [{tag, start, end, msg_idx, step, role, channel}]

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    def at_step(self, n: int) -> TaggedConversation:
        """Truncate to tokens at step <= n."""
        kept = [s for s in self.spans if s["step"] <= n]
        if not kept:
            return TaggedConversation([], {t: torch.zeros(0, dtype=torch.bool) for t in TAGS}, [])
        max_pos = max(s["end"] for s in kept)
        return TaggedConversation(
            tokens=self.tokens[:max_pos],
            masks={t: m[:max_pos].clone() for t, m in self.masks.items()},
            spans=[s for s in self.spans if s["start"] < max_pos],
        )

    def to_span_groups(self, fine: bool = False) -> Dict[str, List[Tuple[int, int]]]:
        """Span groups for attention-mass computation.

        fine=False → coarse groups (control, system, user, assistant, tool_clean, payload, …)
        fine=True  → one group per tag (frame_boundary, developer_tools, assistant_reasoning, …)
        """
        groups: Dict[str, List[Tuple[int, int]]] = {}
        for s in self.spans:
            key = s["tag"] if fine else COARSE.get(s["tag"], s["tag"])
            groups.setdefault(key, []).append((s["start"], s["end"]))
        for k in groups:
            groups[k] = _merge(sorted(groups[k]))
        return groups

    def injection_steps(self) -> List[int]:
        """Steps that contain attack_payload tokens."""
        return sorted({s["step"] for s in self.spans if s["tag"] == "attack_payload"})


# ── Main entry point ───────────────────────────────────────────────────

def tag_conversation(
    formatter: HarmonyFormatter,
    messages: List[Dict[str, Any]],
    tool_descriptions: list,
    *,
    attack_payload: Optional[str] = None,
    attack_prefix: Optional[str] = None,
    attack_suffix: Optional[str] = None,
) -> TaggedConversation:
    """Tag every token in a harmony-rendered conversation."""
    result = formatter.tokenize_with_spans(messages, tool_descriptions)
    token_ids = result.token_ids[:result.generation_prompt_start]
    seq_len = len(token_ids)

    if seq_len == 0:
        return TaggedConversation([], {t: torch.zeros(0, dtype=torch.bool) for t in TAGS}, [])

    raw_spans = result.raw_spans
    content_spans = result.content_spans
    gen_start = result.generation_prompt_start

    tags: List[Optional[str]] = [None] * seq_len
    tags: List[Optional[str]] = [None] * seq_len
    spans: List[dict] = []
    step = 0
    prev_role = None

    for mi, msg in enumerate(messages):
        if msg["role"] != prev_role:
            if prev_role is not None:
                step += 1
            prev_role = msg["role"]
        
        role = msg["role"]
        channel = msg.get("channel")
        recipient = msg.get("recipient")
        rs, cs = raw_spans[mi], content_spans[mi]

        if role == "system":
            _tag_system(token_ids, tags, spans, formatter, rs, mi, step)

        elif role == "user":
            _tag_frame_region(token_ids, tags, spans, rs[0], cs[0], mi, step, role, channel)
            _fill(tags, spans, cs[0], cs[1], "user_instruction", mi, step, role, channel)
            _tag_frame_trailing(token_ids, tags, spans, cs[1], rs[1], mi, step, role, channel)

        elif role == "assistant":
            _tag_frame_region(token_ids, tags, spans, rs[0], cs[0], mi, step, role, channel)
            if channel == "analysis":
                tag = "assistant_reasoning"
            elif channel == "commentary" and recipient:
                tag = "assistant_tool_call"
            elif channel == "commentary":
                tag = "assistant_commentary"
            else:
                tag = "assistant_final"
            _fill(tags, spans, cs[0], cs[1], tag, mi, step, role, channel)
            _tag_frame_trailing(token_ids, tags, spans, cs[1], rs[1], mi, step, role, channel)

        elif role == "tool":
            _tag_frame_region(token_ids, tags, spans, rs[0], cs[0], mi, step, role, channel)
            _tag_tool_content(
                token_ids, tags, spans, formatter, cs,
                mi, step, channel,
                attack_payload, attack_prefix, attack_suffix,
            )
            _tag_frame_trailing(token_ids, tags, spans, cs[1], rs[1], mi, step, role, channel)

    # Generation prompt (incomplete assistant turn at end)
    if gen_start < seq_len:
        _tag_frame_region(token_ids, tags, spans, gen_start, seq_len, -1, step, "gen_prompt", None)

    # Safety net: anything still None gets frame_metadata
    for i in range(seq_len):
        if tags[i] is None:
            tags[i] = "frame_metadata"

    # Build boolean masks
    positions = {t: [] for t in TAGS}
    for i, t in enumerate(tags):
        if t in positions:
            positions[t].append(i)

    masks = {}
    for tag in TAGS:
        idx = positions[tag]
        m = torch.zeros(seq_len, dtype=torch.bool)
        if idx:
            m[torch.tensor(idx, dtype=torch.long)] = True
        masks[tag] = m

    return TaggedConversation(tokens=token_ids, masks=masks, spans=spans)


# ── Granular frame tagging ─────────────────────────────────────────────

def _tag_frame_region(token_ids, tags, spans, start, end, mi, step, role, channel):
    """Tag a framing region (before content) with granular special-token tags.

    State machine:
        <|start|>     → frame_boundary;  following text → frame_role
        <|channel|>   → frame_channel;   following text → frame_channel_name
        <|constrain|> → frame_constrain; following text → frame_constrain_type
        <|message|>   → frame_message
        other specials → their _SPECIAL_TO_TAG entry
        unclassifiable text → frame_metadata
    """
    if start >= end:
        return

    prev_special = None
    i = start
    while i < end:
        tid = token_ids[i]

        if tid in _SPECIAL_TO_TAG:
            _fill(tags, spans, i, i + 1, _SPECIAL_TO_TAG[tid], mi, step, role, channel)
            prev_special = tid
            i += 1
        else:
            # Run of non-special tokens — classify by preceding special
            group_start = i
            while i < end and token_ids[i] not in _SPECIAL_TO_TAG:
                i += 1

            if prev_special == TOKEN_START:
                text_tag = "frame_role"
            elif prev_special == TOKEN_CHANNEL:
                text_tag = "frame_channel_name"
            elif prev_special == TOKEN_CONSTRAIN:
                text_tag = "frame_constrain_type"
            else:
                text_tag = "frame_metadata"

            _fill(tags, spans, group_start, i, text_tag, mi, step, role, channel)


def _tag_frame_trailing(token_ids, tags, spans, start, end, mi, step, role, channel):
    """Tag trailing framing (typically just terminal tokens like <|end|>)."""
    if start >= end:
        return

    i = start
    while i < end:
        tid = token_ids[i]
        if tid in _SPECIAL_TO_TAG:
            _fill(tags, spans, i, i + 1, _SPECIAL_TO_TAG[tid], mi, step, role, channel)
            i += 1
        else:
            group_start = i
            while i < end and token_ids[i] not in _SPECIAL_TO_TAG:
                i += 1
            _fill(tags, spans, group_start, i, "frame_metadata", mi, step, role, channel)


# ── System/developer split ─────────────────────────────────────────────

def _tag_system(token_ids, tags, spans, formatter, rs, mi, step):
    """Handle the system raw span: system_meta message + developer message.

    Harmony renders a system prompt as two messages:
        <|start|>system<|channel|>meta<|message|>…<|end|>
        <|start|>developer<|message|>…<|end|>
    """
    rs_start, rs_end = rs

    start_positions = [i for i in range(rs_start, rs_end) if token_ids[i] == TOKEN_START]
    msg_positions   = [i for i in range(rs_start, rs_end) if token_ids[i] == TOKEN_MESSAGE]
    end_positions   = [i for i in range(rs_start, rs_end) if token_ids[i] == TOKEN_END]

    if len(start_positions) < 2 or len(msg_positions) < 2:
        # Can't split — tag everything as framing
        _tag_frame_region(token_ids, tags, spans, rs_start, rs_end, mi, step, "system", None)
        return

    # ── System meta sub-message ─────────────────────────────────────
    meta_start   = start_positions[0]
    meta_msg_tok = msg_positions[0]

    # Find <|end|> that closes system_meta (must be before developer's <|start|>)
    dev_start = start_positions[1]
    meta_end_tok = None
    for pos in end_positions:
        if meta_msg_tok < pos < dev_start:
            meta_end_tok = pos
            break

    if meta_end_tok is None:
        # No <|end|> found — use developer's <|start|> as boundary
        meta_content_end = dev_start
        meta_raw_end = dev_start
    else:
        meta_content_end = meta_end_tok
        meta_raw_end = meta_end_tok + 1  # include the <|end|> token

    # Framing before meta content
    _tag_frame_region(token_ids, tags, spans, meta_start, meta_msg_tok + 1,
                      mi, step, "system", "meta")
    # Meta content
    _fill(tags, spans, meta_msg_tok + 1, meta_content_end,
          "system_meta", mi, step, "system", "meta")
    # Meta terminal
    if meta_content_end < meta_raw_end:
        _tag_frame_trailing(token_ids, tags, spans, meta_content_end, meta_raw_end,
                            mi, step, "system", "meta")

    # ── Developer sub-message ───────────────────────────────────────
    dev_msg_tok = msg_positions[1]

    # Gap between meta end and developer start (should be empty, but be safe)
    if meta_raw_end < dev_start:
        _fill(tags, spans, meta_raw_end, dev_start, "frame_metadata",
              mi, step, "developer", None)

    # Framing: <|start|>developer<|message|>
    _tag_frame_region(token_ids, tags, spans, dev_start, dev_msg_tok + 1,
                      mi, step, "developer", None)

    dev_content_start = dev_msg_tok + 1
    dev_has_terminal = (rs_end > rs_start and token_ids[rs_end - 1] in TERMINAL_TOKENS)
    dev_content_end = rs_end - 1 if dev_has_terminal else rs_end

    # Split developer content at "# Tools"
    if dev_content_start < dev_content_end:
        content_toks = token_ids[dev_content_start:dev_content_end]
        # Use skip_special_tokens=False so special tokens contribute to char count
        # (avoids silent offset drift if a special token leaks into content)
        text = formatter.tokenizer.decode(content_toks, skip_special_tokens=False)
        tools_pos = text.find("# Tools")

        if tools_pos < 0:
            _fill(tags, spans, dev_content_start, dev_content_end,
                  "developer_instructions", mi, step, "developer", None)
        elif tools_pos == 0:
            _fill(tags, spans, dev_content_start, dev_content_end,
                  "developer_tools", mi, step, "developer", None)
        else:
            split_tok = dev_content_start + _char_to_tok(
                formatter, content_toks, tools_pos, skip_special=False,
            )
            _fill(tags, spans, dev_content_start, split_tok,
                  "developer_instructions", mi, step, "developer", None)
            _fill(tags, spans, split_tok, dev_content_end,
                  "developer_tools", mi, step, "developer", None)

    if dev_has_terminal:
        _tag_frame_trailing(token_ids, tags, spans, rs_end - 1, rs_end,
                            mi, step, "developer", None)


# ── Tool content with injection decomposition ──────────────────────────

def _tag_tool_content(token_ids, tags, spans, formatter, cs, mi, step, channel,
                      payload, prefix, suffix):
    cs_start, cs_end = cs
    if cs_start >= cs_end:
        return

    attack_hits = []
    for attack_str, attack_tag in [
        (prefix,  "attack_prefix"),
        (payload, "attack_payload"),
        (suffix,  "attack_suffix"),
    ]:
        if not attack_str:
            continue
        hit = formatter.find_injection_token_span(token_ids, cs_start, cs_end, attack_str)
        if hit:
            attack_hits.append((hit[0], hit[1], attack_tag))

    if not attack_hits:
        _fill(tags, spans, cs_start, cs_end, "tool_env_data", mi, step, "tool", channel)
        return

    attack_hits.sort()

    # Fix overlapping attack regions (trim earlier span)
    for i in range(len(attack_hits) - 1):
        if attack_hits[i][1] > attack_hits[i + 1][0]:
            attack_hits[i] = (attack_hits[i][0], attack_hits[i + 1][0], attack_hits[i][2])

    cursor = cs_start
    for a_start, a_end, a_tag in attack_hits:
        if cursor < a_start:
            _fill(tags, spans, cursor, a_start, "tool_env_data", mi, step, "tool", channel)
        _fill(tags, spans, a_start, a_end, a_tag, mi, step, "tool", channel)
        cursor = a_end
    if cursor < cs_end:
        _fill(tags, spans, cursor, cs_end, "tool_env_data", mi, step, "tool", channel)


# ── Helpers ────────────────────────────────────────────────────────────

def _fill(tags, spans, start, end, tag, mi, step, role, channel):
    if start >= end:
        return
    for i in range(start, end):
        tags[i] = tag
    spans.append({"tag": tag, "start": start, "end": end,
                  "msg_idx": mi, "step": step, "role": role, "channel": channel})


def _merge(spans):
    if not spans:
        return []
    out = [spans[0]]
    for s, e in spans[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _char_to_tok(formatter, tokens, char_pos, skip_special=False):
    """Character offset → token index via incremental decode.

    Uses skip_special_tokens=False by default so that special tokens
    contribute to the character count (avoids silent offset drift).
    """
    cum = 0
    for i, tok in enumerate(tokens):
        piece = formatter.tokenizer.decode([tok], skip_special_tokens=skip_special)
        cum += len(piece)
        if cum > char_pos:
            return i
    return len(tokens)