"""
token_spans.py
==============

Representation of token-level spans within a multi-turn tool-calling conversation.

AgentDojo conversations consist of messages with roles:
    system    → system prompt + tool definitions
    user      → the user's task instruction  
    assistant → model response (text + tool_calls)
    tool      → tool execution result (THIS is where injection payloads live)

At each inference step, the model sees the full conversation history up to that point.
We need to track which token indices correspond to which message, so we can:
    - Measure attention on payload vs user instruction vs system prompt
    - Do LOO ablation on specific spans
    - Identify the "critical step" where injection enters the context

This module is model-agnostic: it works with any tokenizer. The actual tokenization
is done externally; this just stores and queries the resulting spans.

Usage:
    transcript = ConversationTranscript()
    transcript.add_message("system", "You are a helpful assistant...", token_start=0, token_end=45)
    transcript.add_message("user", "Who is invited to the event?", token_start=45, token_end=62)
    transcript.add_message("assistant", "", token_start=62, token_end=78,
                           tool_calls=[{"name": "get_events", "args": {...}}])
    transcript.add_message("tool", "Event: Networking...\n<INJECTION HERE>", 
                           token_start=78, token_end=145,
                           tool_name="get_events", contains_injection=True,
                           injection_token_start=120, injection_token_end=145)
    
    # Query
    transcript.payload_spans()         # → [(120, 145)]
    transcript.user_instruction_span() # → (45, 62)
    transcript.tool_result_spans()     # → [(78, 145)]
    transcript.get_step(2)             # → messages from step 2
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class MessageSpan:
    """A single message in the conversation with its token-level boundaries."""
    
    role: str                          # "system", "user", "assistant", "tool"
    text: str                          # the raw text content
    token_start: int                   # inclusive start index in the full token sequence
    token_end: int                     # exclusive end index
    
    # Which inference step this message belongs to (0-indexed)
    # Step 0: initial prompt (system + user + first LLM call)
    # Step 1: first tool result + second LLM call
    # etc.
    step: int = 0
    
    # For assistant messages: tool calls made
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # For tool messages: which tool produced this result
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    # Injection tracking
    contains_injection: bool = False
    injection_token_start: Optional[int] = None  # within the full sequence
    injection_token_end: Optional[int] = None
    
    # For tracking which part of the tool output is "data" vs "injection"
    # (the non-injection part is legitimate tool output)
    data_token_start: Optional[int] = None
    data_token_end: Optional[int] = None
    
    @property
    def length(self) -> int:
        return self.token_end - self.token_start
    
    @property
    def span(self) -> Tuple[int, int]:
        return (self.token_start, self.token_end)
    
    @property
    def injection_span(self) -> Optional[Tuple[int, int]]:
        if self.contains_injection and self.injection_token_start is not None:
            return (self.injection_token_start, self.injection_token_end)
        return None


class ConversationTranscript:
    """
    A tokenized multi-turn conversation with span annotations.
    
    Tracks the full message history and provides query methods for
    extracting spans by role, step, or semantic function.
    """
    
    def __init__(self, model_name: str = "", total_tokens: int = 0):
        self.model_name = model_name
        self.total_tokens = total_tokens
        self.messages: List[MessageSpan] = []
        self._step_counter = 0
    
    def add_message(self, role: str, text: str, 
                    token_start: int, token_end: int,
                    step: Optional[int] = None,
                    **kwargs) -> MessageSpan:
        """Add a message to the transcript."""
        if step is None:
            step = self._step_counter
        
        msg = MessageSpan(
            role=role, text=text,
            token_start=token_start, token_end=token_end,
            step=step, **kwargs
        )
        self.messages.append(msg)
        self.total_tokens = max(self.total_tokens, token_end)
        return msg
    
    def advance_step(self):
        """Move to the next inference step."""
        self._step_counter += 1
    
    # ── Query methods ───────────────────────────────────────────────
    
    def by_role(self, role: str) -> List[MessageSpan]:
        return [m for m in self.messages if m.role == role]
    
    def by_step(self, step: int) -> List[MessageSpan]:
        return [m for m in self.messages if m.step == step]
    
    @property
    def num_steps(self) -> int:
        if not self.messages:
            return 0
        return max(m.step for m in self.messages) + 1
    
    def system_spans(self) -> List[Tuple[int, int]]:
        """Token spans of all system messages."""
        return [m.span for m in self.by_role("system")]
    
    def user_instruction_span(self) -> Optional[Tuple[int, int]]:
        """Token span of the first user message (the task instruction)."""
        users = self.by_role("user")
        return users[0].span if users else None
    
    def assistant_spans(self) -> List[Tuple[int, int]]:
        """Token spans of all assistant messages."""
        return [m.span for m in self.by_role("assistant")]
    
    def tool_result_spans(self) -> List[Tuple[int, int]]:
        """Token spans of all tool result messages."""
        return [m.span for m in self.by_role("tool")]
    
    def payload_spans(self) -> List[Tuple[int, int]]:
        """Token spans of all injection payloads (within tool results)."""
        spans = []
        for m in self.messages:
            if m.contains_injection and m.injection_span is not None:
                spans.append(m.injection_span)
        return spans
    
    def clean_data_spans(self) -> List[Tuple[int, int]]:
        """Token spans of tool result data EXCLUDING injection payloads."""
        spans = []
        for m in self.by_role("tool"):
            if m.contains_injection and m.data_token_start is not None:
                spans.append((m.data_token_start, m.data_token_end))
            elif not m.contains_injection:
                spans.append(m.span)
        return spans
    
    def critical_step(self) -> Optional[int]:
        """
        The first inference step where injection content enters the context.
        This is the step AFTER the tool result containing the injection.
        """
        for m in self.messages:
            if m.contains_injection and m.role == "tool":
                # The next assistant turn (same or next step) is the critical one
                return m.step
        return None
    
    def messages_at_critical_step(self) -> List[MessageSpan]:
        """All messages visible at the critical inference step."""
        cs = self.critical_step()
        if cs is None:
            return []
        # Everything up to and including this step
        return [m for m in self.messages if m.step <= cs]
    
    def context_at_step(self, step: int) -> Tuple[int, int]:
        """Token range of the full context visible at a given step."""
        msgs = [m for m in self.messages if m.step <= step]
        if not msgs:
            return (0, 0)
        return (msgs[0].token_start, msgs[-1].token_end)
    
    # ── Aggregation for feature extraction ──────────────────────────
    
    def get_span_groups(self, step: Optional[int] = None) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get all token spans grouped by semantic role, optionally at a specific step.
        
        Returns dict with keys:
            "system", "user", "assistant", "tool_clean", "tool_injected", "payload"
        
        Useful for passing to feature extraction or LOO attribution.
        """
        msgs = self.messages if step is None else [m for m in self.messages if m.step <= step]
        
        groups: Dict[str, List[Tuple[int, int]]] = {
            "system": [],
            "user": [],
            "assistant": [],
            "tool_clean": [],
            "tool_injected": [],
            "payload": [],
        }
        
        for m in msgs:
            if m.role == "system":
                groups["system"].append(m.span)
            elif m.role == "user":
                groups["user"].append(m.span)
            elif m.role == "assistant":
                groups["assistant"].append(m.span)
            elif m.role == "tool":
                if m.contains_injection:
                    groups["tool_injected"].append(m.span)
                    if m.injection_span:
                        groups["payload"].append(m.injection_span)
                else:
                    groups["tool_clean"].append(m.span)
        
        return groups
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "model_name": self.model_name,
            "total_tokens": self.total_tokens,
            "num_messages": len(self.messages),
            "num_steps": self.num_steps,
            "critical_step": self.critical_step(),
            "messages": [
                {
                    "role": m.role,
                    "step": m.step,
                    "token_start": m.token_start,
                    "token_end": m.token_end,
                    "length": m.length,
                    "contains_injection": m.contains_injection,
                    "tool_name": m.tool_name,
                    "tool_calls": m.tool_calls,
                    "injection_span": m.injection_span,
                }
                for m in self.messages
            ],
        }
    
    def __repr__(self):
        lines = [f"ConversationTranscript({self.model_name}, {self.total_tokens} tokens, "
                 f"{len(self.messages)} messages, {self.num_steps} steps)"]
        for m in self.messages:
            inj = " [INJECTION]" if m.contains_injection else ""
            tc = f" → {len(m.tool_calls)} tool calls" if m.tool_calls else ""
            tn = f" ({m.tool_name})" if m.tool_name else ""
            lines.append(f"  step={m.step} {m.role:10s} [{m.token_start}:{m.token_end}]"
                        f"{tn}{tc}{inj}")
        return "\n".join(lines)