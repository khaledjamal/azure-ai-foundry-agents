"""The same chat completion sent twice: once as a raw REST request, once through the openai SDK.

New: not in Microsoft's workshop.

This script sends the same chat question to the same model twice, once as an HTTP request built by hand and once
through the OpenAI Python library. It is a learning companion to the chat completion lab (1-basic-chat-completion.ipynb)
that shows what the SDK does under the hood: the library adds convenience, not magic. The request body is the same, and
the library only adds its own headers (its user agent, for example).

What it does:
  1. REST: `requests.post` to {OPENAI_ENDPOINT}/openai/deployments/{deployment}/chat/completions?api-version=...
     with an `Authorization: Bearer <token>` header and a JSON body.
  2. SDK:  `AzureOpenAI(...).chat.completions.with_raw_response.create(...)`, which builds that same request and
     also exposes the HTTP response's status.
  3. A side-by-side table at the end compares status, latency, tokens and answer length.

Details:
- Both calls go to the same Azure OpenAI deployment with the same messages and settings, signed in with the same
  Entra ID token, so no API key is involved.
- The HTTP log (workshop_utils.http_log) prints both requests and responses in full, headers and JSON bodies, with the
  token redacted, so the two can be compared line by line.
- Exits with 1 unless both calls return an answer.

Settings from .env: OPENAI_ENDPOINT, OPENAI_API_VERSION, MODEL_DEPLOYMENT_NAME. Two small paid calls on
gpt-4o-mini, well under $0.01 per run.

Run from the project root:
    uv run python src/1-models/1-basic-chat-completion-rest-vs-sdk.py [--model gpt-4o-mini]
"""

import argparse
import os
import time

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
SYSTEM_PROMPT = "You are FitChat GPT, a friendly fitness assistant. Remind users you're not a medical professional."
QUESTION = "Give me three quick tips to stay active during a busy workday."
SETTINGS = {"max_tokens": 200, "temperature": 0.7}


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is not set: {name}")
    return value


def call_rest(endpoint, api_version, deployment, token, messages):
    """POST the chat completion request by hand; return (status, seconds, body or None)."""
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": api_version}
    body = {"messages": messages, **SETTINGS}
    start = time.perf_counter()
    response = requests.post(url, params=params, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    seconds = time.perf_counter() - start
    print(f"\nHTTP {response.status_code} in {seconds:.2f} s")
    if not response.ok:
        print(f"Error body: {response.text[:500]}")
        return response.status_code, seconds, None
    return response.status_code, seconds, response.json()


def call_sdk(client, deployment, messages):
    """Same request through the SDK; return (status, seconds, ChatCompletion)."""
    print(f"client.chat.completions.with_raw_response.create(model={deployment!r}, messages=..., **{SETTINGS})")
    start = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(model=deployment, messages=messages, **SETTINGS)
    seconds = time.perf_counter() - start
    print(f"\nHTTP {raw.status_code} in {seconds:.2f} s")
    return raw.status_code, seconds, raw.parse()


def main():
    load_env()
    parser = argparse.ArgumentParser(description="One chat completion through REST and through the openai SDK")
    parser.add_argument("--model", default=os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini")
    args = parser.parse_args()

    endpoint = require_env("OPENAI_ENDPOINT")
    api_version = require_env("OPENAI_API_VERSION")
    print(f"Endpoint: {endpoint}")
    print(f"API version: {api_version}")
    print(f"Model deployment: {args.model}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": QUESTION}]
    print(f"User: {QUESTION}")

    # One Entra ID token provider for both sides: REST puts the token in the header itself, the SDK calls it.
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)

    print_section("1. REST: requests.post")
    rest_status, rest_seconds, rest_body = call_rest(endpoint, api_version, args.model, token_provider(), messages)
    rest_answer, rest_usage = "", {}
    if rest_body:
        rest_answer = rest_body["choices"][0]["message"]["content"] or ""
        rest_usage = rest_body.get("usage", {})
        print(f"Response JSON keys: {list(rest_body)}")
        print(f"\nAssistant: {rest_answer}")

    print_section("2. SDK: openai AzureOpenAI client")
    client = AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=token_provider, api_version=api_version)
    sdk_status, sdk_seconds, completion = call_sdk(client, args.model, messages)
    sdk_answer = completion.choices[0].message.content or ""
    print(f"Parsed into: {type(completion).__module__}.{type(completion).__name__}")
    print(f"\nAssistant: {sdk_answer}")

    print_section("Side by side")
    rows = [
        ("HTTP status", rest_status, sdk_status),
        ("latency (s)", f"{rest_seconds:.2f}", f"{sdk_seconds:.2f}"),
        ("prompt tokens", rest_usage.get("prompt_tokens"), completion.usage.prompt_tokens),
        ("completion tokens", rest_usage.get("completion_tokens"), completion.usage.completion_tokens),
        ("answer length (chars)", len(rest_answer), len(sdk_answer)),
    ]
    print(f"{'':24}{'REST':>12}{'SDK':>12}")
    for name, rest_value, sdk_value in rows:
        print(f"{name:24}{rest_value!s:>12}{sdk_value!s:>12}")
    print("Same prompt tokens show both sides sent the same request; the answers differ because temperature > 0.")

    passed = bool(rest_answer.strip()) and bool(sdk_answer.strip())
    print(f"\nCheck: {'PASS' if passed else 'FAIL'}  both calls returned an answer")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
