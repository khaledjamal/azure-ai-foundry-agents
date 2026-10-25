"""Basic chat completion lab (1-basic-chat-completion.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/1-basic-chat-completion.ipynb

This script makes the simplest call to a model deployed in Azure AI Foundry. It prints every HTTP request and
response, so you can see exactly what the library sends.

What it does:
1. A system message sets the assistant's role (a fitness assistant that reminds users it isn't a medical
   professional), a user question goes in, and the model's answer comes out, with its token counts.
2. A second call fills a prompt template with a user's name and goal, the usual way to personalize prompts.

Connection method:
- Client: the openai package's AzureOpenAI client, pointed straight at the Azure OpenAI endpoint (OPENAI_ENDPOINT).
- Sign-in: an Entra ID token from the Azure CLI login instead of an API key, the way apps on Azure avoid storing
  secrets.
- Twin: 1-basic-chat-completion-project-client.py makes the same calls through the Foundry project SDK, which starts
  from the project and gets the OpenAI client from it.

Run from the project root:
    uv run python src/1-models/1-basic-chat-completion-azure-openai.py
"""

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

FITNESS_SYSTEM_PROMPT = (
    "You are FitChat GPT, a friendly fitness assistant.\n"
    "Always remind users: I'm not a medical professional.\n"
    "Answer with empathy and disclaimers."
)

TRAINER_SYSTEM_TEMPLATE = (
    "You are FitChat GPT, an AI personal trainer for {name}.\n"
    "Your user wants to achieve: {goal}.\n"
    "Remind them you're not a medical professional. Offer friendly advice."
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


def chat(client, model, system_prompt, user_input):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    usage = response.usage
    print(f"(tokens: prompt {usage.prompt_tokens}, completion {usage.completion_tokens})")
    return response.choices[0].message.content


def main():
    print_section("BASIC CHAT COMPLETION - AzureOpenAI")
    load_environment()
    model = require_env("OPENAI_DEPLOYMENT_NAME")
    print(f"Model deployment:  {model}")
    client = create_client()

    print_section("1. Chat with a system prompt")
    question = "How can I start a beginner workout routine at home?"
    print(f"User: {question}")
    answer = chat(client, model, FITNESS_SYSTEM_PROMPT, question)
    print(f"Assistant: {answer}")

    print_section("2. Prompt template with fill-ins")
    system_prompt = TRAINER_SYSTEM_TEMPLATE.format(
        name="Jordan",
        goal="increase muscle tone and endurance",
    )
    question = "What kind of home exercise do you recommend for a busy schedule?"
    print(f"User: {question}")
    answer = chat(client, model, system_prompt, question)
    print(f"Assistant: {answer}")

    print_section("DONE")


if __name__ == "__main__":
    http_log.enable()
    main()
