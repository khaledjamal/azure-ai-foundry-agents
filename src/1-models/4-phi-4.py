"""Phi-4 lab (4-phi-4.ipynb) as a script: Phi-4, Phi-4-reasoning and gpt-4o-mini on the
same prompts.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/4-phi-4.ipynb

This script compares three models on the same three prompts: Microsoft's small Phi-4,
Phi-4-reasoning (which thinks step by step before answering) and gpt-4o-mini. It answers
a practical question: does a smaller model or a reasoning model pay off? In the recorded
run all three got the calculation right, gpt-4o-mini was the fastest at about the same
cost, and Phi-4-reasoning took about 9 times as long for the same answers.

Each model gets three prompts:
  1. the notebook's chat question (a weekly 5K training schedule),
  2. the notebook's "RAG-like" question, answered from a short context snippet,
  3. a pace calculation with one right answer (6:00 per km, 18 seconds per km faster),
     to show what reasoning adds.
A table at the end compares tokens, time and cost per model. Exits with 1 unless every
model answered every prompt.

Changes from the notebook:
- The notebook called a serverless Phi-4 through the azure-ai-inference SDK (retired
  2026-08-26). Here Phi-4 (version 7) and Phi-4-reasoning are deployed on the Foundry
  resource, and all three models are called the same way, like the OpenAI models: the
  openai SDK on the Azure OpenAI v1 endpoint ({OPENAI_ENDPOINT}/openai/v1/) with an
  Entra ID token.
- Phi-4-reasoning gets the system prompt from its model card, which asks for the
  thinking between <think> tags before the answer. On this Azure deployment the tags are
  removed from the reply and the message's `reasoning` field stays empty, so the
  thinking and the answer arrive as one text, printed as is.

Settings from .env: OPENAI_ENDPOINT, PHI_MODEL_DEPLOYMENT_NAME (default Phi-4),
PHI_REASONING_MODEL_DEPLOYMENT_NAME (default Phi-4-reasoning), MODEL_DEPLOYMENT_NAME
(default gpt-4o-mini). Paid: nine small calls, well under $0.01 per run (prices below).

Run from the project root:
    uv run python src/1-models/4-phi-4.py
"""

import os
import re
import time

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
# USD per 1M input / output tokens, Global deployments in East US 2 (Azure Retail Prices
# API, checked 2026-10-18).
PRICES = {"Phi-4": (0.125, 0.50), "Phi-4-reasoning": (0.125, 0.50), "gpt-4o-mini": (0.15, 0.60)}
SYSTEM_PROMPT = (
    "You are a helpful AI assistant, focusing on health and fitness. Remind users that you are not a medical "
    "professional, but can provide general info."
)
CONTEXT_SNIPPET = (
    "Recommended to run 3 times per week and mix with cross-training.\n"
    "Include rest days or active recovery days for muscle repair."
)
PROMPTS = [
    ("chat", SYSTEM_PROMPT, "I'm training for a 5K. Any tips on a weekly workout schedule?"),
    (
        "context",
        (
            "You are a helpful fitness AI. We have some context from the user's knowledge base:\n"
            f"{CONTEXT_SNIPPET}\nPlease use this context to help your answer. If the context doesn't help, say so."
        ),
        "How often should I run weekly to prepare for a 5K?",
    ),
    (
        "pace",
        SYSTEM_PROMPT,
        (
            "I run 5 km in 31 minutes 30 seconds. To finish in 30 minutes, what average pace per km do I need, and "
            "how many seconds per km faster is that than my current pace? "
            "End with one line: 'Answer: <pace>, <seconds>'."
        ),
    ),
]
# The system prompt from Phi-4-reasoning's model card; the lab's own system prompt is
# added after it. The <think> tags it asks for are special tokens, which the Azure
# endpoint strips (it also rejects skip_special_tokens=False).
REASONING_SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly "
    "exploring questions through a systematic thinking process before providing the final precise and accurate "
    "solutions. Please structure your response into two main sections: Thought and Solution using the specified "
    "format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning "
    "process in steps. In the Solution section, based on the various attempts, explorations, and reflections from the "
    "Thought section, systematically present the final solution that you deem correct."
)


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


def cost(model, prompt_tokens, completion_tokens):
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def ask(openai_client, model, system_prompt, question, reasoning):
    if reasoning:
        system_prompt = f"{REASONING_SYSTEM_PROMPT}\n\n{system_prompt}"
    start = time.perf_counter()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        temperature=0.8 if reasoning else 0.3,
        max_tokens=4096 if reasoning else 800,
    )
    seconds = time.perf_counter() - start
    choice = response.choices[0]
    return {
        "seconds": seconds,
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
    """Return the endpoint and the (model, reasoning) pairs: Phi-4, Phi-4-reasoning and
    gpt-4o-mini."""
    endpoint = require_env("OPENAI_ENDPOINT").rstrip("/")
    models = [
        (os.getenv("PHI_MODEL_DEPLOYMENT_NAME") or "Phi-4", False),
        (os.getenv("PHI_REASONING_MODEL_DEPLOYMENT_NAME") or "Phi-4-reasoning", True),
        (os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini", False),
    ]
    print(f"Endpoint: {endpoint}/openai/v1/")
    print(f"Models: {', '.join(model for model, _ in models)}")
    return endpoint, models


def create_openai_client(endpoint):
    """The plain OpenAI client on the Azure OpenAI v1 endpoint, signed in with an Entra
    ID token."""
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    openai_client = OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=token_provider)
    return openai_client


def run_prompts(openai_client, models):
    """Ask every prompt on every model; return {(model, prompt name): result}."""
    results = {}
    for name, system_prompt, question in PROMPTS:
        print_section(f"Prompt '{name}': {question}")
        for model, reasoning in models:
            result = ask(openai_client, model, system_prompt, question, reasoning)
            results[(model, name)] = result
            print_result(name, model, result)
    return results


def print_result(name, model, result):
    print(
        f"\n--- {model}: {result['seconds']:.1f} s, tokens in {result['prompt_tokens']} / out "
        f"{result['completion_tokens']}, finish={result['finish']}"
    )
    print(result["answer"])
    if name == "pace":
        print(f"[pace answer correct: {'yes' if pace_is_correct(result['answer']) else 'no'}]")


def print_comparison(results, models):
    """One row per model: tokens, time and cost over all prompts, and whether the pace
    answer was right."""
    print_section("Comparison (all three prompts)")
    print(f"{'model':18}{'tokens in':>10}{'tokens out':>12}{'seconds':>9}{'cost USD':>11}  pace correct")
    for model, _ in models:
        rows = [results[(model, name)] for name, _, _ in PROMPTS]
        tokens_in = sum(row["prompt_tokens"] for row in rows)
        tokens_out = sum(row["completion_tokens"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        total = cost(model, tokens_in, tokens_out)
        cost_text = f"{total:.5f}" if total is not None else "n/a"
        correct = "yes" if pace_is_correct(results[(model, "pace")]["answer"]) else "no"
        print(f"{model:18}{tokens_in:>10}{tokens_out:>12}{seconds:>9.1f}{cost_text:>11}  {correct}")


def check_results(results):
    """Every model answered every prompt."""
    answered = all(result["answer"].strip() for result in results.values())
    print(f"\nCheck: {'PASS' if answered else 'FAIL'}  every model answered every prompt")
    return answered


def main():
    load_env()
    endpoint, models = load_settings()
    openai_client = create_openai_client(endpoint)
    results = run_prompts(openai_client, models)
    print_comparison(results, models)
    passed = check_results(results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
