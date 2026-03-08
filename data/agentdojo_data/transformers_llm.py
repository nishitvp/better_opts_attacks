"""
HuggingFace Transformers-based LLM for agentdojo — runs in-process
with full PyTorch hook access for attention extraction & profiling.

Usage:
    from transformers_llm import TransformersLLM, HookManager
    from agentdojo.agent_pipeline import AgentPipeline, SystemMessage, InitQuery
    from agentdojo.agent_pipeline import ToolsExecutor, ToolsExecutionLoop

    hooks = HookManager()
    llm = TransformersLLM("openai/gpt-oss-20b", hook_manager=hooks)

    # Register hooks for attention extraction
    hooks.register_attention_hooks(llm.model)
    # — or register custom hooks on any module —
    hooks.register_hook(llm.model.model.layers[0].self_attn, "layer0_attn")

    pipeline = AgentPipeline([
        SystemMessage("You are a helpful assistant."),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ])

    # Run via agentdojo benchmark
    from agentdojo.benchmark import benchmark_suite_with_injections
    results = benchmark_suite_with_injections(pipeline, suite, attack, ...)

    # After each generation, inspect captured data:
    for name, data in hooks.captured.items():
        print(name, [d.shape for d in data])
    hooks.clear()
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  HookManager — attach PyTorch hooks to inspect model internals
# ---------------------------------------------------------------------------


class HookManager:
    """Manages PyTorch forward hooks for capturing intermediate tensors.

    Stores captured tensors in ``self.captured[name]`` as a list of tensors,
    one per forward call.  Call ``clear()`` between benchmark tasks to free memory.

    Example — grab every self-attention output::

        hooks = HookManager()
        hooks.register_attention_hooks(model)  # auto-finds attn layers

    Example — grab a specific module::

        hooks.register_hook(model.model.layers[5].self_attn, "L5_attn")
    """

    def __init__(self) -> None:
        self.captured: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    # -- public API ---------------------------------------------------------

    def register_hook(
        self,
        module: nn.Module,
        name: str,
        capture_output: bool = True,
        capture_input: bool = False,
    ) -> None:
        """Register a forward hook on *module* that stores tensors under *name*."""

        def _hook(_mod: nn.Module, inp: Any, out: Any) -> None:
            if capture_output:
                t = out[0] if isinstance(out, tuple) else out
                self.captured[name].append(t.detach().cpu())
            if capture_input:
                t = inp[0] if isinstance(inp, tuple) else inp
                self.captured[f"{name}_input"].append(t.detach().cpu())

        h = module.register_forward_hook(_hook)
        self._handles.append(h)

    def register_attention_hooks(self, model: nn.Module) -> None:
        """Auto-discover and hook all self-attention modules.

        Works with typical transformer naming conventions
        (``self_attn``, ``attention``, ``attn``).
        """
        count = 0
        for full_name, mod in model.named_modules():
            # Match common attention module names
            short = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
            if short in ("self_attn", "attention", "attn"):
                self.register_hook(mod, full_name)
                count += 1
        logger.info("HookManager: registered attention hooks on %d modules", count)

    def clear(self) -> None:
        """Drop all captured tensors (call between tasks to avoid OOM)."""
        self.captured.clear()

    def remove_all(self) -> None:
        """Remove all registered hooks from the model."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.captured.clear()


# ---------------------------------------------------------------------------
#  agentdojo message ↔ HuggingFace chat format conversion
# ---------------------------------------------------------------------------


def _agentdojo_msgs_to_hf(
    messages: Sequence[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert agentdojo ChatMessage sequence into plain dicts that
    ``tokenizer.apply_chat_template`` understands."""
    hf_msgs: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        text = get_text_content_as_str(msg["content"])

        if role == "system":
            hf_msgs.append({"role": "system", "content": text})

        elif role == "user":
            hf_msgs.append({"role": "user", "content": text})

        elif role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_list = []
                for tc in tool_calls:
                    tc_list.append(
                        {
                            "id": tc.get("id", str(uuid.uuid4())[:8]),
                            "type": "function",
                            "function": {
                                "name": tc["function"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                    )
                assistant_msg["tool_calls"] = tc_list
            hf_msgs.append(assistant_msg)

        elif role == "tool":
            # agentdojo ChatToolResultMessage
            tool_msg = msg  # type: ChatToolResultMessage
            tc = tool_msg.get("tool_call", {})
            hf_msgs.append(
                {
                    "role": "tool",
                    "name": tc.get("function", "unknown"),
                    "tool_call_id": tc.get("id", ""),
                    "content": text if tool_msg.get("error") is None else f"ERROR: {tool_msg['error']}",
                }
            )
        else:
            # Fallback: treat as user
            hf_msgs.append({"role": "user", "content": text})
    return hf_msgs


def _functions_to_hf_tools(tools: Sequence[Function]) -> list[dict[str, Any]]:
    """Convert agentdojo Function objects into the JSON-schema tool format
    understood by HuggingFace chat templates."""
    hf_tools: list[dict[str, Any]] = []
    for fn in tools:
        schema = fn.parameters.model_json_schema()
        hf_tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn.name,
                    "description": fn.description or "",
                    "parameters": schema,
                },
            }
        )
    return hf_tools


# ---------------------------------------------------------------------------
#  Parse harmony-format model output back into agentdojo types
# ---------------------------------------------------------------------------

# Regex patterns for harmony output parsing
_TOOL_CALL_RE = re.compile(
    r"to=functions\.(\S+)\s*"           # function name after "to=functions."
    r"(?:<\|constrain\|>json)?"         # optional constrain marker
    r"<\|message\|>(.*?)<\|call\|>",    # JSON arguments between message/call
    re.DOTALL,
)

_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)


def _parse_harmony_output(raw_text: str) -> ChatAssistantMessage:
    """Parse raw harmony-format text into a ChatAssistantMessage.

    Handles three cases:
    1. One or more tool calls (``to=functions.X ... <|call|>``)
    2. A final text response (``<|channel|>final<|message|>...<|return|>``)
    3. Plain text fallback (no harmony markers found)
    """
    # FunctionCall is a TypedDict with: id, function, args, placeholder_args
    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(raw_text):
        fn_name = match.group(1).strip()
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        tool_calls.append(
            {
                "id": str(uuid.uuid4())[:8],
                "function": fn_name,
                "args": args,
                "placeholder_args": None,
            }
        )

    # Extract text from the "final" channel
    final_match = _FINAL_RE.search(raw_text)
    if final_match:
        text = final_match.group(1).strip()
    elif not tool_calls:
        # Fallback: strip all harmony tokens and use whatever is left
        text = re.sub(r"<\|[^|]*\|>", "", raw_text).strip()
        # Remove the analysis/commentary channel text since it's CoT
        # Try to find a clean response after CoT
        if not text:
            text = ""
    else:
        text = ""

    return ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(text)],
        tool_calls=tool_calls if tool_calls else None,
    )


# ---------------------------------------------------------------------------
#  TransformersLLM — the agentdojo BasePipelineElement
# ---------------------------------------------------------------------------


class TransformersLLM(BasePipelineElement):
    """In-process HuggingFace Transformers LLM for agentdojo.

    Runs the model directly via ``model.generate()`` — no server, no vLLM —
    so you can attach PyTorch hooks for profiling and interpretability.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model identifier or local path (e.g. ``"openai/gpt-oss-20b"``).
    hook_manager:
        Optional :class:`HookManager` for capturing attention weights etc.
    device_map:
        Passed to ``AutoModelForCausalLM.from_pretrained``. Default ``"auto"``.
    torch_dtype:
        Passed to ``from_pretrained``. Default ``"auto"`` (uses model config).
    max_new_tokens:
        Maximum tokens to generate per call.
    temperature:
        Sampling temperature.  0 → greedy.
    model_kwargs:
        Extra kwargs forwarded to ``from_pretrained`` (e.g. ``attn_implementation``).

    Notes
    -----
    * For gpt-oss-20b, the tokenizer's built-in ``chat_template.jinja`` handles
      the harmony response format automatically, including tool definitions.
    * Pass ``attn_implementation="eager"`` in *model_kwargs* if you need raw
      attention matrices (flash-attention fuses the softmax and won't expose them).
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/gpt-oss-20b",
        hook_manager: HookManager | None = None,
        device_map: str = "auto",
        torch_dtype: str | torch.dtype = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        extra = model_kwargs or {}

        logger.info("Loading tokenizer for %s …", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        logger.info("Loading model %s (device_map=%s) …", model_name_or_path, device_map)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **extra,
        )
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hook_manager = hook_manager
        self._model_name = model_name_or_path

        # Build the set of stop-token ids for generation (harmony tokens)
        self._stop_token_ids: list[int] = []
        for tok_str in ("<|return|>", "<|call|>", "<|end|>"):
            ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
            self._stop_token_ids.extend(ids)
        if self.tokenizer.eos_token_id is not None:
            self._stop_token_ids.append(self.tokenizer.eos_token_id)
        # Deduplicate
        self._stop_token_ids = list(set(self._stop_token_ids))

    # -- core agentdojo interface -------------------------------------------

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """Run one LLM generation step.

        Converts the agentdojo messages + available tools into HuggingFace chat
        format, generates with ``model.generate()``, parses the harmony output
        back into agentdojo types, and appends the assistant message.
        """
        messages = list(messages)
        tools = runtime.functions.values()

        # 1) Convert messages and tools to HF format
        hf_messages = _agentdojo_msgs_to_hf(messages)
        hf_tools = _functions_to_hf_tools(tools) if tools else None

        # 2) Tokenize with chat template
        try:
            input_text = self.tokenizer.apply_chat_template(
                hf_messages,
                tools=hf_tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            # Fallback if chat template doesn't support tools kwarg
            input_text = self.tokenizer.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        input_ids = self.tokenizer(
            input_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # 3) Generate
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "eos_token_id": self._stop_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        # 4) Decode only the new tokens
        new_ids = output_ids[0, prompt_len:]
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=False)

        logger.debug("Raw model output:\n%s", raw_output)

        # 5) Parse harmony output into agentdojo message
        assistant_msg = _parse_harmony_output(raw_output)
        messages.append(assistant_msg)

        return query, runtime, env, messages, extra_args

    def __repr__(self) -> str:
        return f"TransformersLLM(model={self._model_name!r})"