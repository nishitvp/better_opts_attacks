#!/usr/bin/env python3
"""
verify_rendering.py
====================

Two tests:

1. Token Rendering: For every AgentDojo task, verify our HarmonyFormatter
   produces identical tokens to canonical openai_harmony (the same library
   the OpenAI API server uses internally). No inference needed.

2. Functional: Run our LocalHarmonyLLM pipeline on tasks, verify the model
   makes tool calls and passes AgentDojo's utility checks.

Usage:
    # Token rendering only (fast, no GPU)
    python verify_rendering.py --rendering-only --model openai/gpt-oss-20b

    # Both tests (needs GPU)
    python verify_rendering.py --model openai/gpt-oss-20b --device cuda:0

    # Specific tasks
    python verify_rendering.py --tasks user_task_0 user_task_3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List
import traceback
import copy

from agentdojo.task_suite import get_suite
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import get_text_content_as_str

from harmony_format import HarmonyFormatter, agentdojo_tools_to_descriptions

logger = logging.getLogger(__name__)

VERSION = "v1.2"
SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to a set of tools. "
    "You can use these tools to help answer the user's questions."
)


# ── Test 1: Token Rendering ────────────────────────────────────────────

def test_token_rendering(model_name: str, suite_name: str, task_ids: List[str]):
    from openai_harmony import (
        load_harmony_encoding, HarmonyEncodingName, Role, Message,
        Conversation, SystemContent, DeveloperContent, ReasoningEffort,
    )
    from transformers import AutoTokenizer

    print("\n" + "="*70)
    print("  Test 1: Token Rendering (ours vs canonical openai_harmony)")
    print("="*70)

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatter = HarmonyFormatter(tokenizer, reasoning_effort="low")

    suite = get_suite(VERSION, suite_name)
    runtime = FunctionsRuntime(suite.tools)
    tool_descs = agentdojo_tools_to_descriptions(runtime)

    passed, total = 0, 0

    for task_id in task_ids:
        if task_id not in suite.user_tasks:
            continue
        task = suite.user_tasks[task_id]
        prompt = task.prompt if hasattr(task, "prompt") else str(task)

        # Our formatter
        ours = formatter.render(
            [{"role": "system", "content": SYSTEM_MESSAGE},
             {"role": "user", "content": prompt}],
            tool_descs,
        )

        # Canonical openai_harmony
        sys_c = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
        dev_c = DeveloperContent.new().with_instructions(SYSTEM_MESSAGE)
        if tool_descs:
            dev_c = dev_c.with_function_tools(tool_descs)
        canonical = enc.render_conversation_for_completion(
            Conversation.from_messages([
                Message.from_role_and_content(Role.SYSTEM, sys_c),
                Message.from_role_and_content(Role.DEVELOPER, dev_c),
                Message.from_role_and_content(Role.USER, prompt),
            ]),
            Role.ASSISTANT,
        )

        total += 1
        if ours == canonical:
            passed += 1
            print(f"  ✓ {task_id} ({len(ours)} tokens)")
        else:
            print(f"  ✗ {task_id}: ours={len(ours)} canonical={len(canonical)}")
            for i in range(min(len(ours), len(canonical))):
                if ours[i] != canonical[i]:
                    s = max(0, i-3)
                    print(f"    First diff at {i}: ours={ours[s:i+3]} canonical={canonical[s:i+3]}")
                    break

    print(f"\n  Result: {passed}/{total} identical")
    return passed == total


# ── Test 2: Functional ─────────────────────────────────────────────────

def test_functional(model_name: str, suite_name: str, task_ids: List[str], device: str):
    from inference_runner import LocalHarmonyLLM, build_pipeline

    print("\n" + "="*70)
    print("  Test 2: Functional (run pipeline, check utility)")
    print("="*70)

    llm = LocalHarmonyLLM(model_name=model_name, device=device)
    pipeline = build_pipeline(llm, system_message=SYSTEM_MESSAGE)

    suite = get_suite(VERSION, suite_name)
    results = []

    for task_id in task_ids:
        if task_id not in suite.user_tasks:
            continue

        task = suite.user_tasks[task_id]
        prompt = task.PROMPT

        vectors = suite.get_injection_vector_defaults()
        MARKER = "___MARKER___"
        injections = {v: f"{MARKER}{v}{MARKER}" for v in vectors}
        env = suite.load_and_inject_default_environment(injections)
        if hasattr(task, "init_environment"):
            env = task.init_environment(env)
        runtime = FunctionsRuntime(suite.tools)

        original_env = copy.deepcopy(env)
        t0 = time.time()
        try:
            _, _, post_env, messages, _ = pipeline.query(prompt, runtime, env)
            elapsed = time.time() - t0

            tools = [tc.function for m in messages if m["role"] == "assistant" and m.get("tool_calls")
                     for tc in m["tool_calls"]]
            turns = sum(1 for m in messages if m["role"] == "tool")

            final = ""
            for m in reversed(messages):
                if m["role"] == "assistant" and not m.get("tool_calls"):
                    final = get_text_content_as_str(m["content"])
                    break

            try:
                utility = task.utility(final, original_env, post_env)
            except Exception:
                traceback.print_exc()
                utility = None

            status = "✓" if utility else "✗"
            print(f"  {status} {task_id}: {turns} tool turns, "
                  f"{len(tools)} calls, utility={utility}, {elapsed:.1f}s")

            results.append({
                "task_id": task_id, "utility": utility,
                "tool_turns": turns, "tool_calls": tools,
                "has_response": bool(final),
            })

        except Exception as e:
            traceback.print_exc()
            print(f"  ✗ {task_id}: FAILED ({e})")
            results.append({"task_id": task_id, "utility": None, "error": str(e)})

    # Summary
    n = len(results)
    util_ok = sum(1 for r in results if r.get("utility"))
    has_calls = sum(1 for r in results if r.get("tool_calls"))
    print(f"\n  Summary: {n} tasks")
    print(f"    Utility passed: {util_ok}/{n}")
    print(f"    Made tool calls: {has_calls}/{n}")

    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--suite", default="workspace")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rendering-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["transformers", "torch"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    suite = get_suite(VERSION, args.suite)
    task_ids = args.tasks or list(suite.user_tasks.keys())
    print(f"Suite: {args.suite}, Tasks: {len(task_ids)}")

    # rendering_ok = test_token_rendering(args.model, args.suite, task_ids)

    # if args.rendering_only:
    #     sys.exit(0 if rendering_ok else 1)

    results = test_functional(args.model, args.suite, task_ids, args.device)

    Path("verification_results.json").write_text(
        json.dumps({"functional": results}, indent=2, default=str)
    )
    print("\nSaved to verification_results.json")


if __name__ == "__main__":
    main()