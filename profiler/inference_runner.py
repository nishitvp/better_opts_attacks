"""
inference_runner.py
===================

Local GPT-OSS LLM that plugs into AgentDojo's pipeline infrastructure.

Instead of reimplementing tool execution, message handling, and evaluation,
we implement BasePipelineElement and let AgentDojo's pipeline handle everything:
    SystemMessage → InitQuery → OurLLM → ToolsExecutionLoop[ToolsExecutor, OurLLM]

Our LLM element does ONLY:
    1. Convert AgentDojo ChatMessages → harmony tokens (via openai_harmony)
    2. Run forward pass through local model
    3. Parse output back to ChatAssistantMessage

Profiling data (token spans, features) accumulates on the LLM element for
later retrieval.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import SystemMessage, InitQuery
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, ToolsExecutionLoop
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.functions_runtime import FunctionsRuntime, FunctionCall, Function
from agentdojo.task_suite import get_suite
from agentdojo.types import ChatMessage, ChatAssistantMessage, get_text_content_as_str

from token_spans import ConversationTranscript
from harmony_format import (
    HarmonyFormatter, STOP_TOKEN_IDS, agentdojo_tools_to_descriptions,
)
from feature_hooks import FeatureCapture, FeatureConfig
from conversation_tagger import tag_conversation                        # ← ADDED

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = ""


# ── ChatMessage conversion (used by both LLM element and verification) ──


def chat_messages_to_dicts(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert AgentDojo ChatMessage TypedDicts to simple dicts for harmony rendering.

    AgentDojo ChatMessages have content as list[MessageContentBlock].
    Our harmony formatter expects {"role": str, "content": str, ...}.
    """
    import json as _json
    result = []
    for msg in messages:
        role = msg["role"]

        if role == "system":
            text = get_text_content_as_str(msg["content"])
            result.append({"role": "system", "content": text})

        elif role == "user":
            text = get_text_content_as_str(msg["content"])
            result.append({"role": "user", "content": text})

        elif role == "assistant":
            if msg["content"] is not None:
                text = get_text_content_as_str(msg["content"])
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    result.append({
                        "role": "assistant",
                        "content": tc.args if isinstance(tc.args, str) else _json.dumps(tc.args),
                        "channel": "commentary",
                        "recipient": f"functions.{tc.function}",
                    })
            else:
                result.append({"role": "assistant", "content": text, "channel": "final"})

        elif role == "tool":
            tc = msg["tool_call"]
            result.append({"role": "tool", "name": tc.function, "content": msg["content"]})

    return result

# ── Local LLM as BasePipelineElement ────────────────────────────────────

class LocalHarmonyLLM(BasePipelineElement):
    """AgentDojo pipeline element that runs GPT-OSS locally with harmony format.

    Drop-in replacement for OpenAILLM. Receives ChatMessages from the pipeline,
    renders them to harmony tokens, generates, parses the output back to
    ChatAssistantMessage, and returns it to the pipeline.

    Profiling data (token spans per call, features) is accumulated on the
    instance for later retrieval.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 512,
        reasoning_effort: str = "low",
        feature_config: Optional[FeatureConfig] = None,
        **model_kwargs,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info("Loading model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=device, **model_kwargs,
        )
        self.model.eval()

        self.formatter = HarmonyFormatter(self.tokenizer, reasoning_effort=reasoning_effort)
        self.feature_config = feature_config or FeatureConfig()
        self.feature_capture = FeatureCapture(self.model, self.feature_config)

        # Accumulated profiling data across pipeline calls
        self.call_log: List[Dict[str, Any]] = []

        # Prefix KV cache — single-use per run_task() call.
        # _prefix_cache_ids_ref: the stored prefix (set once per context, kept across runs).
        # _prefix_cache_ids:     the "armed" copy (cleared after first hit; restored by
        #                        rearm_prefix_cache() before each run_task()).
        # This prevents the cache from being applied to later generate() calls within the
        # same pipeline run (calls 2+ see a longer context and should not use the cache).
        self._prefix_cache_ids_ref: Optional[List[int]] = None
        self._prefix_cache_ids: Optional[List[int]] = None
        self._prefix_cache_kv = None  # DynamicCache on GPU
        self._prefix_cache_hits: int = 0
        self._prefix_cache_misses: int = 0

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env=None,
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Any, Sequence[ChatMessage], dict]:
        """Called by AgentDojo's pipeline. Renders, generates, parses."""

        # Convert AgentDojo ChatMessages → simple dicts for harmony rendering
        simple_msgs = chat_messages_to_dicts(messages)

        # Convert AgentDojo Function objects → harmony ToolDescriptions
        tool_descs = agentdojo_tools_to_descriptions(runtime)

        # Render to tokens
        token_ids = self.formatter.render(simple_msgs, tool_descs)

        # Generate
        gen_ids = self._generate(token_ids)

        # Parse
        gen = self.formatter.parse_generation(gen_ids)

        # Log for profiling
        self.call_log.append({
            "prompt_tokens": len(token_ids),
            "gen_tokens": len(gen_ids),
            "prompt_token_ids": list(token_ids),
            "gen_token_ids": list(gen_ids),
            "tool_calls": [
                {"name": tc.function_name, "args": tc.arguments}
                for tc in gen.tool_calls
            ],
            "has_final": gen.final_text is not None,
            "messages_in": len(messages),
        })

        # Convert back to AgentDojo ChatAssistantMessage
        assistant_msg = self._generation_to_chat_message(gen)

        # Append to message history (AgentDojo convention)
        messages = list(messages) + [assistant_msg]

        return query, runtime, env, messages, extra_args

    # ── Prefix KV cache management ───────────────────────────────────

    def set_prefix_cache(self, prefix_ids: List[int], past_key_values) -> None:
        """Store a KV prefix cache for this context.

        The cache is SINGLE-USE per run_task() call: _prefix_cache_ids is cleared
        (disarmed) after the first generate() hit, preventing the cache from being
        applied to later calls in the same pipeline run (calls 2+ have a longer
        context and must not use a stale prefix cache — doing so causes divergence).

        Call rearm_prefix_cache() immediately before each run_task() to re-enable
        the cache for that call only.
        """
        self._prefix_cache_ids_ref = list(prefix_ids)
        self._prefix_cache_ids = list(prefix_ids)   # armed for first use
        self._prefix_cache_kv = past_key_values
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        n_layers = len(getattr(past_key_values, "layers", []))
        seq_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else "?"
        logger.info("Prefix KV cache set: %d prefix tokens, %d layers, seq_len=%s",
                    len(prefix_ids), n_layers, seq_len)

    def rearm_prefix_cache(self) -> None:
        """Re-arm the cache for the next run_task() call.

        Must be called immediately before each run_task() so the cache applies to
        only the FIRST matching generate() call in that pipeline run.
        """
        if self._prefix_cache_ids_ref is not None:
            self._prefix_cache_ids = list(self._prefix_cache_ids_ref)

    def clear_prefix_cache(self) -> None:
        if self._prefix_cache_ids_ref is not None:
            logger.info("Prefix KV cache cleared: %d hits, %d misses",
                        self._prefix_cache_hits, self._prefix_cache_misses)
        self._prefix_cache_ids_ref = None
        self._prefix_cache_ids = None
        self._prefix_cache_kv = None

    # ── Internal ────────────────────────────────────────────────────

    def _generate(self, prompt_ids: List[int]) -> List[int]:
        # Check for an armed prefix cache match.  Disarms itself after first hit so
        # subsequent calls in the same run_task() use standard (uncached) generation.
        if self._prefix_cache_ids is not None:
            n = len(self._prefix_cache_ids)
            if (len(prompt_ids) > n
                    and list(prompt_ids[:n]) == self._prefix_cache_ids):
                self._prefix_cache_hits += 1
                self._prefix_cache_ids = None  # disarm: only cache first hit per run
                logger.debug("_generate: cache HIT (prompt_len=%d, prefix_len=%d)",
                             len(prompt_ids), n)
                return self._generate_from_cache(prompt_ids)
            logger.debug("_generate: cache MISS (prompt_len=%d, prefix_len=%d, match=%s)",
                         len(prompt_ids), n,
                         "too_short" if len(prompt_ids) <= n else "prefix_mismatch")

        self._prefix_cache_misses += 1
        input_ids = torch.tensor([prompt_ids], device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=STOP_TOKEN_IDS,
            )
        return out[0][len(prompt_ids):].tolist()

    def _generate_from_cache(self, prompt_ids: List[int]) -> List[int]:
        """Generate using the cached prefix KV state.

        Strategy to restore the cache after generate() mutates it:
        - Sliding-window layers: deepcopy before generate, restore after (~5ms for 3MB)
        - Full-attention (global) layers: crop() back to prefix_len (O(1))

        Uses _prefix_cache_ids_ref (the permanent reference) rather than
        _prefix_cache_ids, which is cleared (disarmed) before this call.
        """
        import copy

        prefix_len = len(self._prefix_cache_ids_ref)
        suffix_ids = prompt_ids[prefix_len:]

        # Snapshot just the sliding-window layers (tiny: 128 tokens × 8 heads per layer)
        sliding_snapshots: Dict[int, Any] = {}
        for i, layer in enumerate(self._prefix_cache_kv.layers):
            if getattr(layer, "is_sliding", False):
                sliding_snapshots[i] = copy.deepcopy(layer)

        input_ids = torch.tensor([suffix_ids], device=self.device)
        position_ids = torch.arange(
            prefix_len, prefix_len + len(suffix_ids),
            device=self.device, dtype=torch.long,
        ).unsqueeze(0)
        attention_mask = torch.ones(
            1, prefix_len + len(suffix_ids),
            device=self.device, dtype=torch.long,
        )

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                past_key_values=self._prefix_cache_kv,
                position_ids=position_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=STOP_TOKEN_IDS,
            )

        # Restore cache state so the next call starts from the same prefix.
        for i, layer in enumerate(self._prefix_cache_kv.layers):
            if i in sliding_snapshots:
                # Restore sliding layer from snapshot (copy tensors to avoid aliasing)
                snap = sliding_snapshots[i]
                for attr in ("keys", "values"):
                    val = getattr(snap, attr, None)
                    setattr(layer, attr, val.clone() if isinstance(val, torch.Tensor) else val)
                for attr in ("cumulative_length",):
                    if hasattr(snap, attr):
                        setattr(layer, attr, getattr(snap, attr))
            else:
                # Full-attention layer: crop back to prefix_len (O(1))
                layer.crop(prefix_len)

        # out[0] = [suffix_ids..., generated_ids...]
        return out[0][len(suffix_ids):].tolist()

    def _generation_to_chat_message(self, gen) -> ChatAssistantMessage:
        """Convert HarmonyGeneration → AgentDojo ChatAssistantMessage."""
        tool_calls = None
        if gen.has_tool_calls:
            tool_calls = [
                FunctionCall(
                    function=tc.function_name,
                    args=tc.arguments,
                    id=tc.id,
                )
                for tc in gen.tool_calls
            ]
            # Tool call turns have no content (matches OpenAI API behavior)
            content = None
        else:
            content = [{"type": "text", "content": gen.final_text or ""}]

        return ChatAssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )


# ── Pipeline builder ────────────────────────────────────────────────────

def build_pipeline(llm: LocalHarmonyLLM, system_message: str = DEFAULT_SYSTEM_MESSAGE) -> AgentPipeline:
    """Build an AgentDojo pipeline using our local LLM.

    Mirrors AgentPipeline.from_config() for the no-defense case:
        SystemMessage → InitQuery → LLM → ToolsExecutionLoop[ToolsExecutor, LLM]
    """
    return AgentPipeline([
        SystemMessage(system_message),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ])


# ── Task runner (thin wrapper around AgentDojo's benchmark flow) ────────

@dataclass
class TaskResult:
    messages: List[ChatMessage] = field(default_factory=list)
    final_response: str = ""
    evaluation: Dict[str, Any] = field(default_factory=dict)
    transcript: Optional[ConversationTranscript] = None
    wall_time: float = 0.0
    call_log: List[Dict[str, Any]] = field(default_factory=list)
    # Preserved so profiling.py can use them downstream
    runtime: Optional[FunctionsRuntime] = None
    attack_string: str = ""


def run_task(
    pipeline: AgentPipeline,
    llm: LocalHarmonyLLM,
    suite_name: str,
    user_task_id: str,
    injection_task_id: str,
    attack_string: str,
    profile_steps: str = "critical",
    version: str = "v1.2",
) -> TaskResult:
    """Run a single AgentDojo task through our pipeline.

    Uses AgentDojo's own infrastructure for everything:
    - Suite/environment/injection setup
    - Pipeline execution (tool calls, message management)
    - Evaluation (utility + security)

    We add only: profiling passes after execution.
    """
    import copy

    t0 = time.time()

    suite = get_suite(version, suite_name)
    user_task = suite.user_tasks[user_task_id]
    injection_task = suite.injection_tasks[injection_task_id]

    # Inject attack into environment
    defaults = suite.get_injection_vector_defaults()
    injections = {v: attack_string for v in defaults}
    env = suite.load_and_inject_default_environment(injections)
    if hasattr(user_task, "init_environment"):
        env = user_task.init_environment(env)

    # Deepcopy before pipeline mutates env (needed for evaluation)
    original_env = copy.deepcopy(env)

    runtime = FunctionsRuntime(suite.tools)
    user_prompt = user_task.PROMPT

    # Clear LLM call log for this task
    llm.call_log.clear()

    # Run the pipeline — AgentDojo handles everything
    _, _, post_env, messages, _ = pipeline.query(user_prompt, runtime, env)

    # Extract final response from last non-tool-call assistant message
    final_response = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and not msg.get("tool_calls"):
            if msg["content"] is not None:
                final_response = get_text_content_as_str(msg["content"])
            break

    # Evaluate via AgentDojo — utility(final, original_env, post_env)
    evaluation = {}
    try:
        evaluation["utility"] = user_task.utility(final_response, original_env, post_env)
    except Exception as e:
        evaluation["utility"] = None
        evaluation["utility_error"] = str(e)
    try:
        sec = injection_task.security(final_response, original_env, post_env)
        evaluation["security"] = sec
        evaluation["injection_success"] = sec
    except Exception as e:
        evaluation["security"] = None
        evaluation["security_error"] = str(e)

    # Build transcript for profiling (from the messages AgentDojo produced)
    transcript = _build_transcript_from_messages(messages, llm, runtime, attack_string)

    return TaskResult(
        messages=list(messages),
        final_response=final_response,
        evaluation=evaluation,
        transcript=transcript,
        wall_time=time.time() - t0,
        call_log=list(llm.call_log),
        runtime=runtime,
        attack_string=attack_string,
    )


def _build_transcript_from_messages(
    messages: Sequence[ChatMessage],
    llm: LocalHarmonyLLM,
    runtime: FunctionsRuntime,
    attack_string: str,
) -> ConversationTranscript:
    """Build a ConversationTranscript from the final message list.

    Tokenizes the full conversation once and records content spans.
    """
    simple_msgs = chat_messages_to_dicts(messages)
    tool_descs = agentdojo_tools_to_descriptions(runtime)
    tok = llm.formatter.tokenize_with_spans(simple_msgs, tool_descs)

    transcript = ConversationTranscript(model_name=llm.model_name)
    step = 0

    # Map content spans to messages. Harmony inserts system+developer,
    # so there's an offset.
    n_spans = len(tok.content_spans)
    n_msgs = len(simple_msgs)
    has_dev = n_spans > n_msgs
    span_offset = 2 if has_dev else 1

    # Record system (merged system+developer)
    if has_dev and n_spans >= 2:
        transcript.add_message(
            role="system", text=simple_msgs[0].get("content", ""),
            token_start=tok.content_spans[0][0],
            token_end=tok.content_spans[1][1], step=0,
        )
    elif n_spans >= 1:
        transcript.add_message(
            role="system", text=simple_msgs[0].get("content", ""),
            token_start=tok.content_spans[0][0],
            token_end=tok.content_spans[0][1], step=0,
        )

    # Record remaining messages
    for i, msg in enumerate(simple_msgs[1:], start=1):
        si = span_offset + (i - 1)
        if si >= n_spans:
            break
        span = tok.content_spans[si]
        role = msg["role"]

        if role == "tool":
            inj_span = None
            if attack_string:
                inj_span = llm.formatter.find_injection_token_span(
                    tok.token_ids, span[0], span[1], attack_string)
            step += 1
            transcript.add_message(
                role="tool", text=msg.get("content", ""),
                token_start=span[0], token_end=span[1], step=step,
                tool_name=msg.get("name"),
                contains_injection=inj_span is not None,
                injection_token_start=inj_span[0] if inj_span else None,
                injection_token_end=inj_span[1] if inj_span else None,
            ) 
        elif role == "assistant":
            transcript.add_message(
                role="assistant", text=msg.get("content", ""),
                token_start=span[0], token_end=span[1], step=step,
            )
        elif role == "user":
            transcript.add_message(
                role="user", text=msg.get("content", ""),
                token_start=span[0], token_end=span[1], step=step,
            )

    return transcript