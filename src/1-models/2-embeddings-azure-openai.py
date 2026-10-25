"""Text embeddings part of the embeddings lab (2-embeddings.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb

This script turns text into embeddings: lists of numbers (vectors) that capture meaning, so texts about similar things
end up close together. Embeddings are the building block of search by meaning and of RAG (the next lab stores them in
Azure AI Search).

What it does:
1. Embeds three short health phrases.
2. Embeds a user question with a prompt template in front.
3. Prints each vector's length and first values.

Connection method:
- Client: the openai package's AzureOpenAI client, created directly with the Azure OpenAI endpoint (OPENAI_ENDPOINT).
- Sign-in: an Entra ID token from the Azure CLI login instead of an API key.
- Twin: 2-embeddings-project-client.py makes the same calls through the Foundry project SDK.

Run from the project root:
    uv run python src/1-models/2-embeddings-azure-openai.py
"""

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

TEXT_PHRASES = [
    "An apple a day keeps the doctor away 🍎",
    "Quick 15-minute HIIT workout routine 🏋️",
    "Mindful breathing exercises 🧘",
]

TEMPLATE_SYSTEM = (
    "You are HealthFitGPT, a fitness guidance model.\n"
    "Please focus on healthy advice and disclaim you're not a doctor.\n\n"
    "User message:"
)


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_environment():
    env_path = load_env(required=True)
    print(f"Loaded .env from: {env_path}")


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is not set: {name}")
    return value


def create_client():
    endpoint = require_env("OPENAI_ENDPOINT")
    api_version = require_env("OPENAI_API_VERSION")

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

    print("Connection method: AzureOpenAI (direct)")
    print(f"Endpoint:          {endpoint}")
    print(f"API version:       {api_version}")
    return client


def sample(vector):
    return f"[{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-2]:.4f}, {vector[-1]:.4f}]"


def main():
    print_section("TEXT EMBEDDINGS - AzureOpenAI")
    load_environment()
    model = require_env("EMBEDDING_MODEL_DEPLOYMENT_NAME")
    print(f"Model deployment:  {model}")
    client = create_client()

    print_section("1. Embed health-themed phrases")
    response = client.embeddings.create(model=model, input=TEXT_PHRASES)
    for i, item in enumerate(response.data):
        print(f"Sentence {i}: '{TEXT_PHRASES[i]}'")
        print(f"  Embedding length={len(item.embedding)}")
        print(f"  Sample: {sample(item.embedding)}")
    print(f"(tokens: {response.usage.total_tokens})")

    print_section("2. Embed user text with a prompt template in front")
    user_text = "Can you suggest a quick home workout for busy moms?"
    response = client.embeddings.create(model=model, input=[f"{TEMPLATE_SYSTEM} {user_text}"])
    vector = response.data[0].embedding
    print(f"User text: {user_text}")
    print(f"Embedding length={len(vector)}")
    print(f"First few dims: {[round(v, 6) for v in vector[:8]]}")
    print(f"(tokens: {response.usage.total_tokens})")

    print_section("DONE")


if __name__ == "__main__":
    http_log.enable()
    main()
