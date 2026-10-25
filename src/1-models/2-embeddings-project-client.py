"""Text embeddings part of the embeddings lab (2-embeddings.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb

This script makes the same embedding calls as its twin, 2-embeddings-azure-openai.py, but through the Azure AI Foundry
project SDK. Embeddings are lists of numbers (vectors) that capture meaning, so texts about similar things end up close
together; they are the building block of search by meaning and of RAG.

What it does:
1. Embeds three short health phrases.
2. Embeds a user question with a prompt template in front.
3. Prints each vector's length and first values.

Connection method:
- Client: AIProjectClient from azure-ai-projects, created with the Foundry project endpoint, connects to the project
  and hands back a ready AzureOpenAI client through get_openai_client().
- Sign-in: an Entra ID token from the Azure CLI login instead of an API key.

Run from the project root:
    uv run python src/1-models/2-embeddings-project-client.py
"""

import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

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


def get_project_endpoint():
    direct_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if direct_endpoint:
        return direct_endpoint.rstrip("/")
    services_endpoint = require_env("AZURE_AI_SERVICES_ENDPOINT").rstrip("/")
    project_name = require_env("AZURE_AI_PROJECT_NAME")
    return f"{services_endpoint}/api/projects/{project_name}"


def create_clients():
    project_endpoint = get_project_endpoint()
    api_version = require_env("OPENAI_API_VERSION")

    project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
    client = project_client.get_openai_client(api_version=api_version)

    print("Connection method: AIProjectClient.get_openai_client()")
    print(f"Project endpoint:  {project_endpoint}")
    print(f"Model endpoint:    {client.base_url}  (chosen by the project client)")
    print(f"API version:       {api_version}")
    return project_client, client


def sample(vector):
    return f"[{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-2]:.4f}, {vector[-1]:.4f}]"


def main():
    print_section("TEXT EMBEDDINGS - AIProjectClient")
    load_environment()
    model = require_env("EMBEDDING_MODEL_DEPLOYMENT_NAME")
    print(f"Model deployment:  {model}")
    project_client, client = create_clients()

    with project_client:
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
