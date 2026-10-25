"""Image generation part of the embeddings lab (2-embeddings.ipynb, section 5) as a script, on gpt-image-2.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb

This script generates an image from a text prompt with gpt-image-2 in Azure AI Foundry and saves it as a PNG file. You
choose the quality, which sets the price: from under a cent at low quality to about 20 cents at high.

What it does:
1. Sends the notebook's prompt through the openai SDK's `images.generate` on the Azure OpenAI v1 endpoint
   ({OPENAI_ENDPOINT}/openai/v1/).
2. Gets the image back as base64 (gpt-image models don't return a URL) and saves it as a PNG named with a timestamp.
3. Checks the file: exits with 1 unless a valid PNG of the requested size was saved.

Changes from the notebook:
- Model: the notebook used DALL-E 3, which was retired on Azure on 2026-03-04; this version moves to its successor,
  gpt-image-2.
- Sign-in: an Entra ID token from the Azure CLI login instead of the notebook's API key.
- It replaces an earlier script, old-dalle3_image_generation.py.

Settings from .env: OPENAI_ENDPOINT, IMAGE_MODEL_DEPLOYMENT_NAME (default gpt-image-2). Paid: one image
per run; at OpenAI's published gpt-image rates about $0.006 at low quality (the default), $0.05 medium, $0.21 high
(the Azure price for gpt-image-2 wasn't listed when checked on 2026-10-18; see Cost Management after a run).

Run from the project root:
    uv run python src/1-models/2-embeddings-image-generation.py [--quality low|medium|high]
        [--size 1024x1024] [--output-dir generated-images] [--prompt "..."]
"""

import argparse
import base64
import os
import struct
import time
from datetime import datetime
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
# The notebook's prompt (section 5).
DEFAULT_PROMPT = "A watercolor painting of fresh fruits and vegetables arranged in a heart shape"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is not set: {name}")
    return value


def png_size(data):
    """Return (width, height) from a PNG's IHDR header, or None if the bytes aren't a PNG."""
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one health-themed image with gpt-image-2")
    parser.add_argument("--model", default=os.getenv("IMAGE_MODEL_DEPLOYMENT_NAME") or "gpt-image-2")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--size", default="1024x1024", help="WIDTHxHEIGHT (default 1024x1024)")
    parser.add_argument("--output-dir", type=Path, default=Path("generated-images"))
    return parser.parse_args()


def create_client():
    """Return the plain OpenAI client on the Azure OpenAI v1 endpoint, signed in with an Entra ID token."""
    endpoint = require_env("OPENAI_ENDPOINT").rstrip("/")
    base_url = f"{endpoint}/openai/v1/"
    print(f"Endpoint: {base_url}")
    # The v1 endpoint takes the plain OpenAI client; the token provider is called for each request.
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    client = OpenAI(base_url=base_url, api_key=token_provider)
    return client


def generate_image(client, args):
    """Step 1: send the prompt through images.generate and return the result. The HTTP log shows the request."""
    print(f"Model deployment: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.size}, quality: {args.quality}")
    start = time.perf_counter()
    result = client.images.generate(
        model=args.model, prompt=args.prompt, size=args.size, quality=args.quality, output_format="png", n=1
    )
    seconds = time.perf_counter() - start
    print(f"Done in {seconds:.1f} s")
    return result


def print_details(result):
    """Print the revised prompt (if the model rewrote it) and the token usage."""
    image = result.data[0]
    if getattr(image, "revised_prompt", None):
        print(f"Revised prompt: {image.revised_prompt}")
    if result.usage:
        print(
            f"Usage: input {result.usage.input_tokens} tokens, output {result.usage.output_tokens} tokens, "
            f"total {result.usage.total_tokens}"
        )


def save_image(image, output_dir):
    """Step 2: decode the base64 image and save it as a PNG named with a timestamp; return (path, size) or (None, ...)."""
    data = base64.b64decode(image.b64_json) if image.b64_json else b""
    size = png_size(data)
    if not data:
        return None, size
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = output_dir / f"health-image-{datetime.now():%Y%m%d-%H%M%S}.png"
    saved.write_bytes(data)
    if size:
        print(f"\nSaved: {saved} ({len(data):,} bytes, {size[0]}x{size[1]} px)")
    else:
        print(f"\nSaved: {saved}")
    return saved, size


def check_image(saved, size, requested_size):
    """Step 3: a PNG of the requested size (WIDTHxHEIGHT) must have been saved."""
    expected = tuple(int(part) for part in requested_size.lower().split("x"))
    passed = saved is not None and size == expected
    print(f"\nCheck: {'PASS' if passed else 'FAIL'}  a {requested_size} PNG was returned and saved")
    return passed


def main():
    load_env()
    args = parse_args()
    client = create_client()
    result = generate_image(client, args)
    print_details(result)
    saved, size = save_image(result.data[0], args.output_dir)
    passed = check_image(saved, size, args.size)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
