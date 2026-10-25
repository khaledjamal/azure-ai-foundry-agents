"""DeepSeek lab (5-deep-seek-r1.ipynb) as a script, on DeepSeek-V4-Flash: the same prompts with thinking off and on.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/5-deep-seek-r1.ipynb

This script runs DeepSeek-V4-Flash on the same three prompts twice, with thinking off and on, to show what step-by-step
reasoning adds and what it costs. In the recorded run both modes got the calculation right; with thinking, that answer
took 4.8 s instead of 1.4 s, and the three prompts cost $0.003 instead of $0.002.

Each prompt runs twice, thinking off and on:
  1. the notebook's travel planning question (5 cultural days in Kyoto),
  2. the notebook's technical question (a PostgreSQL database at 10k transactions per second),
  3. the pace calculation from the Phi-4 lab, with one right answer (6:00 per km, 18 seconds per km faster).
It prints each answer, the thinking for the "on" calls, and a table comparing tokens, time and cost of the two modes.
Exits with 1 unless every call answered and every "on" call returned its thinking.

Changes from the notebook:
- Model: the notebook called a serverless DeepSeek-R1, now retired on Azure (Microsoft lists DeepSeek-V4-Pro as its
  replacement); this script uses the cheaper DeepSeek-V4-Flash (GA), deployed on the Foundry resource.
- SDK: the notebook used the azure-ai-inference SDK (retired 2026-08-26); here the calls go through the Azure OpenAI
  v1 endpoint ({OPENAI_ENDPOINT}/openai/v1/) with an Entra ID token.
- Thinking: the notebook cut the reasoning out of the reply with a <think> regex. V4-Flash answers without thinking
  by default; with reasoning_effort="high" it thinks first and returns the thinking in the message's separate
  `reasoning_content` field, so no regex is needed. (The DeepSeek API's own `thinking` request field is rejected by
  the Azure endpoint.)

Settings from .env: OPENAI_ENDPOINT, DEEPSEEK_MODEL_DEPLOYMENT_NAME (default DeepSeek-V4-Flash). Paid:
six calls, about $0.01 per run (prices below).

Run from the project root:
    uv run python src/1-models/5-deep-seek-v4-flash.py
"""

import os
import re
import time

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
# USD per 1M input / output tokens, DeepSeek-V4-Flash Global deployment in East US 2 (Azure Retail Prices API,
# checked 2026-10-18). DeepSeek-R1 was 1.35 / 5.40.
PRICE_IN, PRICE_OUT = 0.19, 0.51
MAX_TOKENS = 4096
# (name, system prompt or None, user message); the first two are the notebook's prompts.
PROMPTS = [
    (
        "trip",
        "You are a travel expert. Provide detailed plans with rationale.",
        "Plan a 5-day cultural trip to Kyoto in April. Include hidden gems and safety considerations.",
    ),
    (
        "technical",
        None,
        (
            "How can I optimize a PostgreSQL database handling 10k transactions/second?\n"
            "Consider indexing strategies, hardware requirements, and query optimization. "
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    ),
    (
        "pace",
        None,
        (
            "I run 5 km in 31 minutes 30 seconds. To finish in 30 minutes, what average pace per km do I need, and "
            "how many seconds per km faster is that than my current pace? "
            "End with one line: 'Answer: <pace>, <seconds>'."
        ),
    ),
]
MODES = [("off", None), ("on", "high")]  # (thinking, reasoning_effort)


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is not set: {name}")
    return value


def pace_is_correct(answer):
    """True if the final 'Answer:' line names a 6:00 (360 s) pace and 18 seconds."""
    lines = [line for line in answer.splitlines() if "answer" in line.lower()]
    last = lines[-1] if lines else answer[-200:]
    pace_ok = "6:00" in last or re.search(r"\b360\b", last) is not None
    return pace_ok and re.search(r"\b18\b", last) is not None


def ask(client, model, system_prompt, question, reasoning_effort):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": question})
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
    )
    seconds = time.perf_counter() - start
    choice = response.choices[0]
    thinking = (choice.message.model_extra or {}).get("reasoning_content") or ""
    return {
        "seconds": seconds,
        "thinking": thinking.strip(),
        "answer": (choice.message.content or "").strip(),
        "finish": choice.finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_settings():
    endpoint = require_env("OPENAI_ENDPOINT").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL_DEPLOYMENT_NAME") or "DeepSeek-V4-Flash"
    print(f"Endpoint: {endpoint}/openai/v1/")
    print(f"Model: {model}, thinking off (default) and on (reasoning_effort=high)")
    return endpoint, model


def create_client(endpoint):
    """The plain OpenAI client on the Azure OpenAI v1 endpoint, signed in with an Entra ID token."""
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    client = OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=token_provider)
    return client


def run_prompts(client, model):
    """Ask every prompt with thinking off and on; return {(prompt name, mode): result}."""
    results = {}
    for name, system_prompt, question in PROMPTS:
        print_section(f"Prompt '{name}': {question}")
        for mode, reasoning_effort in MODES:
            result = ask(client, model, system_prompt, question, reasoning_effort)
            results[(name, mode)] = result
            print_result(name, mode, result)
    return results


def print_result(name, mode, result):
    print(
        f"\n--- thinking {mode}: {result['seconds']:.1f} s, tokens in {result['prompt_tokens']} / out "
        f"{result['completion_tokens']}, finish={result['finish']}"
    )
    if result["thinking"]:
        print(f"[thinking, {len(result['thinking'].split())} words]")
        print(result["thinking"])
        print("[answer]")
    print(result["answer"])
    if name == "pace":
        print(f"[pace answer correct: {'yes' if pace_is_correct(result['answer']) else 'no'}]")


def print_comparison(results):
    """One row per mode: tokens, time and cost over all prompts, and whether the pace answer was right."""
    print_section("Comparison (all three prompts)")
    print(f"{'thinking':10}{'tokens in':>10}{'tokens out':>12}{'seconds':>9}{'cost USD':>11}  pace correct")
    for mode, _ in MODES:
        rows = [results[(name, mode)] for name, _, _ in PROMPTS]
        tokens_in = sum(row["prompt_tokens"] for row in rows)
        tokens_out = sum(row["completion_tokens"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1_000_000
        correct = "yes" if pace_is_correct(results[("pace", mode)]["answer"]) else "no"
        print(f"{mode:10}{tokens_in:>10}{tokens_out:>12}{seconds:>9.1f}{cost:>11.5f}  {correct}")


def check_results(results):
    """Every call answered, and every thinking-on call returned its thinking."""
    checks = [
        ("every call answered", all(result["answer"] for result in results.values())),
        (
            "every thinking-on call returned its thinking",
            all(results[(name, "on")]["thinking"] for name, _, _ in PROMPTS),
        ),
    ]
    print()
    for check, passed in checks:
        print(f"Check: {'PASS' if passed else 'FAIL'}  {check}")
    return all(passed for _, passed in checks)


def main():
    load_env()
    endpoint, model = load_settings()
    client = create_client(endpoint)
    results = run_prompts(client, model)
    print_comparison(results)
    passed = check_results(results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
