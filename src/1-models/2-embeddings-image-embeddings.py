"""Image embeddings part of the embeddings lab (2-embeddings.ipynb, section 4) as a script, on Cohere embed-v-4-0.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb

This script shows that one model can place an image and text in the same space of meaning: Cohere's embed-v-4-0 in
Azure AI Foundry puts images and text in one vector space. That is the basis of searching images with words, for
example finding scans by a description. It goes one step further than the notebook's "print a vector".

What it does:
  1. POST {AZURE_AI_SERVICES_ENDPOINT}/models/images/embeddings  a hand X-ray image as a base64 data URL, input_type
     document
  2. POST {AZURE_AI_SERVICES_ENDPOINT}/models/embeddings         three short captions, input_type query
  3. Ranks the captions by cosine similarity to the image; the X-ray caption must come first.
Exits with 1 unless both calls succeed and the X-ray caption ranks first.

Changes from the notebook:
- Model: the notebook embedded hand-xray.png with Cohere-embed-v3-english; this version uses embed-v-4-0.
- No SDK: the notebook used the azure-ai-inference SDK, a beta that Microsoft retired on 2026-08-26, and the openai
  SDK has no image embeddings, so this version calls the Foundry models REST API directly with requests.
- Sign-in: an Entra ID token instead of an API key.
- It replaces an earlier script, old-image_embeddings_cohere.py.

Settings from .env: AZURE_AI_SERVICES_ENDPOINT, IMAGE_EMBEDDING_MODEL_DEPLOYMENT_NAME (default
embed-v-4-0), IMAGE_EMBEDDING_API_VERSION (default 2024-05-01-preview). Paid: two small calls, well under $0.01 per
run (text about $0.12 per 1M tokens; the image is billed as tokens too, printed after the call).

Run from the project root:
    uv run python src/1-models/2-embeddings-image-embeddings.py [--image PATH]
"""

import argparse
import base64
import math
import os
import time
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "hand-xray.png"
# The first caption describes the image; the check expects it to rank first.
CAPTIONS = [
    "An X-ray image of a human hand",
    "A bowl of fresh fruit and vegetables",
    "A person jogging in a park",
]


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is not set: {name}")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Embed an image and captions with Cohere embed-v-4-0, rank captions")
    parser.add_argument("--model", default=os.getenv("IMAGE_EMBEDDING_MODEL_DEPLOYMENT_NAME") or "embed-v-4-0")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="PNG image (default: hand-xray.png)")
    return parser.parse_args()


def load_settings():
    endpoint = require_env("AZURE_AI_SERVICES_ENDPOINT").rstrip("/")
    api_version = os.getenv("IMAGE_EMBEDDING_API_VERSION") or "2024-05-01-preview"
    return endpoint, api_version


def read_image(path):
    image_bytes = path.read_bytes()
    print(f"Image: {path.name} ({len(image_bytes):,} bytes)")
    return image_bytes


def sign_in():
    """Get an Entra ID token for Azure AI services from the Azure CLI login (no API key)."""
    get_token = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    token = get_token()
    return token


def post_embeddings(url, api_version, token, body):
    """POST one embeddings request and return the response JSON. The HTTP log shows the request and response."""
    start = time.perf_counter()
    response = requests.post(
        url, params={"api-version": api_version}, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=60
    )
    seconds = time.perf_counter() - start
    print(f"HTTP {response.status_code} in {seconds:.2f} s")
    if not response.ok:
        raise RuntimeError(f"Embeddings call failed: HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def show_vector(label, vector):
    sample = f"[{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-2]:.4f}, {vector[-1]:.4f}]"
    print(f"  {label}: length {len(vector)}, {sample}")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def embed_image(endpoint, api_version, token, model, image_bytes):
    """Step 1: embed the image, sent as a base64 data URL with input_type document; return its vector."""
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    body = {"model": model, "input": [{"image": data_url}], "input_type": "document"}
    result = post_embeddings(f"{endpoint}/models/images/embeddings", api_version, token, body)
    image_vector = result["data"][0]["embedding"]
    show_vector("image", image_vector)
    print(f"  usage: {result.get('usage')}")
    return image_vector


def embed_captions(endpoint, api_version, token, model):
    """Step 2: embed the captions with input_type query; return their vectors in the order of CAPTIONS."""
    body = {"model": model, "input": CAPTIONS, "input_type": "query"}
    result = post_embeddings(f"{endpoint}/models/embeddings", api_version, token, body)
    # Each item carries the index of its input; sorting by it lines the vectors up with CAPTIONS.
    items = sorted(result["data"], key=lambda item: item["index"])
    caption_vectors = [item["embedding"] for item in items]
    for caption, vector in zip(CAPTIONS, caption_vectors, strict=True):
        show_vector(repr(caption), vector)
    print(f"  usage: {result.get('usage')}")
    return caption_vectors


def rank_captions(image_vector, caption_vectors):
    """Step 3: rank the captions by cosine similarity to the image, closest first, and print the ranking."""
    scores = [
        (cosine_similarity(image_vector, vector), caption)
        for caption, vector in zip(CAPTIONS, caption_vectors, strict=True)
    ]
    ranked = sorted(scores, reverse=True)
    for rank, (score, caption) in enumerate(ranked, start=1):
        print(f"  {rank}. {score:.4f}  {caption}")
    return ranked


def check_ranking(ranked):
    """The caption that describes the image (CAPTIONS[0]) must rank first."""
    passed = ranked[0][1] == CAPTIONS[0]
    print(f"\nCheck: {'PASS' if passed else 'FAIL'}  the X-ray caption is the closest to the image")
    return passed


def main():
    load_env()
    args = parse_args()
    endpoint, api_version = load_settings()
    print(f"Endpoint: {endpoint}/models")
    print(f"Model deployment: {args.model}")
    image_bytes = read_image(args.image)
    token = sign_in()

    print("\n1. Image embedding")
    image_vector = embed_image(endpoint, api_version, token, args.model, image_bytes)

    print("\n2. Caption embeddings")
    caption_vectors = embed_captions(endpoint, api_version, token, args.model)

    print("\n3. Captions ranked by cosine similarity to the image")
    ranked = rank_captions(image_vector, caption_vectors)

    passed = check_ranking(ranked)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
