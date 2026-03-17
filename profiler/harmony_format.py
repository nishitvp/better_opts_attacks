"""
harmony_format.py
=================

Renders AgentDojo conversations into GPT-OSS harmony tokens and parses
model output back into structured tool calls.

Replicates what OpenAI's API server does when it receives a
chat.completions.create() call with role="system" messages and tools=[...]:
  - system message content → harmony developer instructions
  - tools JSON schemas → harmony TypeScript namespace in developer message
  - harmony system message (identity/meta) is auto-generated

Uses the official openai_harmony library for rendering and parsing.

Dependencies: pip install openai-harmony
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

logger = logging.getLogger(__name__)

# ── Harmony special token IDs ───────────────────────────────────────────

TOKEN_START     = 200006
TOKEN_END       = 200007
TOKEN_MESSAGE   = 200008
TOKEN_CHANNEL   = 200005
TOKEN_CONSTRAIN = 200003
TOKEN_RETURN    = 200002  # stop: generation complete
TOKEN_CALL      = 200012  # stop: tool dispatch

HARMONY_SPECIAL_IDS = frozenset({
    TOKEN_START, TOKEN_END, TOKEN_MESSAGE,
    TOKEN_CHANNEL, TOKEN_CONSTRAIN, TOKEN_RETURN, TOKEN_CALL,
})
STOP_TOKEN_IDS = [TOKEN_RETURN, TOKEN_CALL]
TERMINAL_TOKENS = frozenset({TOKEN_END, TOKEN_RETURN, TOKEN_CALL})


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class HarmonyToolCall:
    function_name: str
    arguments: dict
    raw_recipient: str
    id: str = ""

@dataclass
class HarmonyGeneration:
    tool_calls: List[HarmonyToolCall] = field(default_factory=list)
    final_text: Optional[str] = None
    analysis_text: Optional[str] = None
    commentary_text: Optional[str] = None
    stopped_by: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

@dataclass
class TokenizationResult:
    token_ids: List[int]
    content_spans: List[Tuple[int, int]]  # content-only (between <|message|> and terminal)
    raw_spans: List[Tuple[int, int]]      # full message including framing tokens
    generation_prompt_start: int = 0


# ── Tool conversion ────────────────────────────────────────────────────

def agentdojo_tools_to_descriptions(runtime) -> List[ToolDescription]:
    """Convert FunctionsRuntime.functions → openai_harmony ToolDescriptions.

    Uses Function.description (short+long from docstring_parser) and
    Function.parameters.model_json_schema() — same data the OpenAI API
    receives when AgentDojo calls chat.completions.create(tools=...).
    """
    return [
        ToolDescription.new(
            func.name,
            func.description,
            parameters=func.parameters.model_json_schema(),
        )
        for func in runtime.functions.values()
    ]


# ── Content span extraction ────────────────────────────────────────────

def extract_content_span(
    token_ids: List[int], raw_start: int, raw_end: int
) -> Tuple[int, int]:
    """Find content tokens within a message span (after <|message|>,
    before terminal token). Returns empty span if no <|message|> found."""
    message_pos = None
    for i in range(raw_start, raw_end):
        if token_ids[i] == TOKEN_MESSAGE:
            message_pos = i
    if message_pos is None:
        return (raw_start, raw_start)
    content_start = message_pos + 1
    content_end = raw_end
    if raw_end > raw_start and token_ids[raw_end - 1] in TERMINAL_TOKENS:
        content_end = raw_end - 1
    return (min(content_start, content_end), content_end)


# ── Formatter ───────────────────────────────────────────────────────────

class HarmonyFormatter:
    """Renders AgentDojo conversations to harmony tokens and parses output.

    Replicates the OpenAI API server's behavior:
    - role="system" content → developer instructions
    - tools → TypeScript namespace in developer message
    - harmony system message (identity/reasoning/channels) is auto-generated
    """

    def __init__(self, hf_tokenizer, reasoning_effort: str = "low"):
        self.tokenizer = hf_tokenizer
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._effort = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM,
                        "high": ReasoningEffort.HIGH}.get(reasoning_effort, ReasoningEffort.LOW)

    # ── Rendering ───────────────────────────────────────────────────

    def _to_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_descriptions: List[ToolDescription],
    ) -> Conversation:
        """Convert AgentDojo-format messages to openai_harmony Conversation.

        AgentDojo sends exactly what the OpenAI Chat Completions API expects:
          {"role": "system", "content": "instructions"}
          {"role": "user", "content": "query"}
          {"role": "assistant", "content": "text", "tool_calls": [...]}
          {"role": "tool", "content": "result", "tool_call_id": "..."}

        The OpenAI server maps this to harmony as:
          system  → SystemContent (meta) + DeveloperContent (instructions + tools)
          user    → User message
          assistant text → assistant/final channel
          assistant tool_call → assistant/commentary channel with recipient
          tool result → tool role with author name
        """
        harmony_msgs = []
        system_instructions = ""

        # First pass: extract system content (there's typically one system msg)
        for msg in messages:
            if msg["role"] == "system":
                system_instructions = msg.get("content", "")
                break

        # Always emit system meta + developer
        harmony_msgs.append(
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(self._effort),
            )
        )
        dev_content = DeveloperContent.new()
        if system_instructions:
            dev_content = dev_content.with_instructions(system_instructions)
        if tool_descriptions:
            dev_content = dev_content.with_function_tools(tool_descriptions)
        harmony_msgs.append(
            Message.from_role_and_content(Role.DEVELOPER, dev_content)
        )

        # Second pass: everything except system
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                continue  # already handled

            elif role == "user":
                harmony_msgs.append(
                    Message.from_role_and_content(Role.USER, content)
                )

            elif role == "assistant":
                channel = msg.get("channel", "final")
                recipient = msg.get("recipient")
                hm = Message.from_role_and_content(
                    Role.ASSISTANT, content
                ).with_channel(channel)
                if recipient:
                    hm = hm.with_recipient(recipient)
                    hm = hm.with_content_type("<|constrain|> json")
                harmony_msgs.append(hm)

            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                if not tool_name.startswith("functions."):
                    tool_name = f"functions.{tool_name}"
                text = self._harmony_content_to_text(content)
                harmony_msgs.append(
                    Message.from_author_and_content(
                        Author.new(Role.TOOL, tool_name), text,
                    ).with_channel("commentary")
                )

        return Conversation.from_messages(harmony_msgs)

    def render(
        self,
        messages: List[Dict[str, Any]],
        tool_descriptions: List[ToolDescription],
    ) -> List[int]:
        """Render conversation to token IDs with generation prompt appended."""
        convo = self._to_conversation(messages, tool_descriptions)
        return self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    # ── Tokenization with spans ─────────────────────────────────────

    def tokenize_with_spans(
        self,
        messages: List[Dict[str, Any]],
        tool_descriptions: List[ToolDescription],
    ) -> TokenizationResult:
        """Tokenize and compute per-message content-only spans."""
        if not messages:
            return TokenizationResult([], [], [])

        raw_spans = []
        prev_len = 0
        for i in range(len(messages)):
            partial = self.render(messages[:i + 1], tool_descriptions)
            clean_len = self._strip_gen_prompt(partial)
            raw_spans.append((prev_len, clean_len))
            prev_len = clean_len

        full = self.render(messages, tool_descriptions)
        content_spans = [
            extract_content_span(full, s, e) for s, e in raw_spans
        ]
        return TokenizationResult(full, content_spans, raw_spans, prev_len)

    def tokenize_incremental(
        self,
        prev_messages: List[Dict[str, Any]],
        new_message: Dict[str, Any],
        tool_descriptions: List[ToolDescription],
        prev_token_count: Optional[int] = None,
    ) -> Tuple[List[int], Tuple[int, int], Tuple[int, int]]:
        """Tokenize after appending one message. Returns (tokens, raw_span, content_span)."""
        if prev_token_count is None:
            prev_token_count = self._strip_gen_prompt(
                self.render(prev_messages, tool_descriptions)
            ) if prev_messages else 0

        all_msgs = prev_messages + [new_message]
        full = self.render(all_msgs, tool_descriptions)
        curr_clean = self._strip_gen_prompt(full)

        raw_span = (prev_token_count, curr_clean)
        content_span = extract_content_span(full, raw_span[0], raw_span[1])
        return full, raw_span, content_span

    def _strip_gen_prompt(self, token_ids: List[int]) -> int:
        """Find where the trailing <|start|>assistant generation prompt begins."""
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == TOKEN_START:
                if not any(token_ids[j] == TOKEN_MESSAGE for j in range(i + 1, len(token_ids))):
                    return i
                break
        return len(token_ids)

    # ── Parsing ─────────────────────────────────────────────────────

    def _harmony_content_to_text(self, content) -> str:
        if isinstance(content, str):
            return content

        # Common Harmony case: content is a list like [TextContent(text="...")]
        if isinstance(content, (list, tuple)):
            parts = []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(part))
            return "".join(parts)

        text = getattr(content, "text", None)
        if isinstance(text, str):
            return text

        return str(content)


    def parse_generation(
        self, generated_token_ids: List[int], call_id_prefix: str = "call",
    ) -> HarmonyGeneration:
        """Parse generated token IDs via openai_harmony. No regex."""
        if not generated_token_ids:
            return HarmonyGeneration()

        stopped_by = None
        tokens = generated_token_ids
        if tokens[-1] == TOKEN_CALL:
            stopped_by = "call"
            tokens = tokens[:-1]
        elif tokens[-1] == TOKEN_RETURN:
            stopped_by = "return"
            tokens = tokens[:-1]

        try:
            parsed = self.encoding.parse_messages_from_completion_tokens(
                tokens, role=Role.ASSISTANT
            )
        except Exception as e:
            logger.error("parse failed: %s", e)
            return HarmonyGeneration(
                final_text=self.encoding.decode_utf8(generated_token_ids),
                stopped_by=stopped_by,
            )

        result = HarmonyGeneration(stopped_by=stopped_by)
        call_counter = 0

        for msg in parsed:
            channel = getattr(msg, "channel", None)
            recipient = getattr(msg, "recipient", None)
            content = self._harmony_content_to_text(getattr(msg, "content", None))

            # In Harmony, recipient is the right signal for a tool call.
            if channel == "commentary" and recipient:
                func_name = recipient.removeprefix("functions.")
                try:
                    args = json.loads(content) if content.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": content}

                call_counter += 1
                result.tool_calls.append(
                    HarmonyToolCall(
                        function_name=func_name,
                        arguments=args,
                        raw_recipient=recipient,
                        id=f"{call_id_prefix}_{call_counter}",
                    )
                )

            elif channel == "analysis":
                if content:
                    result.analysis_text = (
                        result.analysis_text + "\n" + content
                        if result.analysis_text else content
                    )

            elif channel == "commentary":
                if content:
                    result.commentary_text = (
                        result.commentary_text + "\n" + content
                        if result.commentary_text else content
                    )

            elif channel == "final":
                if content:
                    result.final_text = (
                        result.final_text + "\n" + content
                        if result.final_text else content
                    )
    
        return result
    # ── Message helpers ─────────────────────────────────────────────
    # These produce the dicts that _to_conversation() expects.
    # They match what AgentDojo's pipeline elements produce.

    @staticmethod
    def system_message(content: str) -> Dict[str, Any]:
        return {"role": "system", "content": content}

    @staticmethod
    def user_message(content: str) -> Dict[str, Any]:
        return {"role": "user", "content": content}

    @staticmethod
    def assistant_analysis_message(cot: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": cot, "channel": "analysis"}

    @staticmethod
    def assistant_tool_call_message(name: str, arguments: dict) -> Dict[str, Any]:
        return {"role": "assistant", "content": json.dumps(arguments),
                "channel": "commentary", "recipient": f"functions.{name}"}

    @staticmethod
    def assistant_final_message(text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": text, "channel": "final"}

    @staticmethod
    def tool_result_message(name: str, content: str) -> Dict[str, Any]:
        return {"role": "tool", "name": name, "content": content}

    def generation_to_messages(self, gen: HarmonyGeneration) -> List[Dict[str, Any]]:
        """Convert parsed generation back to message dicts for conversation history."""
        msgs = []
        if gen.analysis_text:
            msgs.append(self.assistant_analysis_message(gen.analysis_text))
        if gen.commentary_text:
            msgs.append({"role": "assistant", "content": gen.commentary_text, "channel": "commentary"})
        for tc in gen.tool_calls:
            msgs.append(self.assistant_tool_call_message(tc.function_name, tc.arguments))
        if gen.final_text is not None:
            msgs.append(self.assistant_final_message(gen.final_text))
        return msgs

    # ── Injection span detection ────────────────────────────────────

    def find_injection_token_span(
        self, full_token_ids: List[int],
        content_start: int, content_end: int,
        attack_string: str,
    ) -> Optional[Tuple[int, int]]:
        """Find attack_string within a content span. Exact match first, char-ratio fallback."""
        attack_ids = self.tokenizer.encode(attack_string, add_special_tokens=False)
        content_ids = full_token_ids[content_start:content_end]

        # Exact subsequence
        for i in range(len(content_ids) - len(attack_ids) + 1):
            if content_ids[i:i + len(attack_ids)] == attack_ids:
                return (content_start + i, content_start + i + len(attack_ids))

        # Character-ratio fallback
        content_text = self.tokenizer.decode(content_ids, skip_special_tokens=True)
        pos = content_text.find(attack_string)
        if pos >= 0:
            ratio = pos / max(len(content_text), 1)
            est_start = content_start + int(ratio * len(content_ids))
            est_len = max(1, int(len(attack_string) / max(len(content_text), 1) * len(content_ids)))
            return (est_start, min(est_start + est_len, content_end))

        return None

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)