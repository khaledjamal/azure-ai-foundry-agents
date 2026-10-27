"""AutoGen multi-agent RAG lab (2-autogen-multi-agent-rag.ipynb) as a script, on AutoGen
0.7.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/2-autogen-multi-agent-rag.ipynb

This script has two AI agents share a task with AutoGen, Microsoft's open-source
framework for multi-agent apps. Splitting the work this way keeps the answer close to
the retrieved tips.

What it does (the two agents take turns in a RoundRobinGroupChat):
1. RetrieverAgent calls a local retrieve_tips tool to find the most relevant health
   tips.
2. ResponderAgent writes the answer, instructed to use only those tips.

Changes from the notebook:
- Uses the Azure OpenAI deployments of the Foundry project (OPENAI_ENDPOINT,
  MODEL_DEPLOYMENT_NAME, EMBEDDING_MODEL_DEPLOYMENT_NAME) with an Entra ID token from
  the Azure CLI login, instead of GitHub Models with a GITHUB_TOKEN.
- No hand-written model_info: the notebook declared function_calling False while giving
  the retriever a tool.
- retrieve_tips searches by meaning (embeddings + cosine similarity, top 3) instead of
  keywords; the notebook matched any query word as a substring, so "a" or "i" matched
  every tip.
- The retriever passes the tool result on word for word (reflect_on_tool_use=False) and
  the responder may only use those tips, so the answer stays grounded in them.
- The chat stops after the responder's answer (the notebook's max_turns=4 ran the
  retriever again).
- The final result is a TaskResult (no .content); the script reports its stop reason and
  exits 1 if the responder never answered.

Run from the project root:
    uv run python src/4-frameworks/2-autogen-multi-agent-rag.py
"""

import asyncio
import math
import os
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, SourceMatchTermination
from autogen_agentchat.messages import ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
USER_QUERY = "I'm very busy but want to stay fit. What quick exercises can I do?"
TOP_TIPS = 3

HEALTH_TIPS = [
    {"id": "tip1", "content": "Do a 10-minute HIIT workout to boost your metabolism.", "source": "Fitness Guru"},
    {
        "id": "tip2",
        "content": "Take a brisk 15-minute walk to clear your mind and improve circulation.",
        "source": "Health Coach",
    },
    {
        "id": "tip3",
        "content": "Stretch for 5 minutes every hour if you're sitting at a desk.",
        "source": "Wellness Expert",
    },
    {
        "id": "tip4",
        "content": "Incorporate strength training twice a week for overall fitness.",
        "source": "Personal Trainer",
    },
    {"id": "tip5", "content": "Drink water regularly to stay hydrated during workouts.", "source": "Nutritionist"},
]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def make_retrieve_tips(embed):
    """Embed the tips once; the returned tool embeds each query and returns the closest
    tips."""
    tip_vectors = embed([tip["content"] for tip in HEALTH_TIPS])
    print(f"Embedded {len(HEALTH_TIPS)} health tips.")

    def retrieve_tips(query: str) -> str:
        """Return the health tips closest in meaning to the user's question."""
        query_vector = embed([query])[0]
        ranked = sorted(
            zip(HEALTH_TIPS, (cosine(query_vector, v) for v in tip_vectors), strict=True),
            key=lambda p: p[1],
            reverse=True,
        )
        print(
            f"  [tool] retrieve_tips(query={query!r}) similarity: "
            + ", ".join(f"{tip['id']} {score:.2f}" for tip, score in ranked)
        )
        return "\n".join(f"Source: {tip['source']} => {tip['content']}" for tip, _ in ranked[:TOP_TIPS])

    return retrieve_tips


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


def create_clients():
    deployment = require_env("MODEL_DEPLOYMENT_NAME")
    embedding_deployment = require_env("EMBEDDING_MODEL_DEPLOYMENT_NAME")
    endpoint = require_env("OPENAI_ENDPOINT")
    api_version = require_env("OPENAI_API_VERSION")
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    print(f"Azure OpenAI endpoint: {endpoint}")
    print(f"Chat deployment:       {deployment} (API version {api_version})")
    print(f"Embedding deployment:  {embedding_deployment}")

    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=deployment,
        model=deployment,  # AutoGen looks up the model's capabilities (e.g. tool calling) by this name
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )
    openai_client = AzureOpenAI(
        azure_endpoint=endpoint, api_version=api_version, azure_ad_token_provider=token_provider
    )

    def embed(texts):
        return [
            item.embedding for item in openai_client.embeddings.create(model=embedding_deployment, input=texts).data
        ]

    return model_client, embed


def create_team(model_client, retrieve_tips):
    retriever = AssistantAgent(
        name="RetrieverAgent",
        system_message=(
            "You are a smart retrieval agent. Your task is to fetch relevant fitness and health tips for the "
            "user's question. Call the tool 'retrieve_tips' once with the user's question."
        ),
        model_client=model_client,
        tools=[retrieve_tips],
        reflect_on_tool_use=False,  # pass the retrieved tips on word for word
    )
    responder = AssistantAgent(
        name="ResponderAgent",
        system_message=(
            "You are a friendly fitness coach. Craft a fun and engaging response to the user's question using only "
            "the health tips the RetrieverAgent provided, and name each tip's source. Do not add exercises, tips or "
            "facts that are not in those tips. Say briefly that this is not medical advice."
        ),
        model_client=model_client,
    )
    # Stop as soon as the responder has answered; MaxMessageTermination is a safety
    # limit.
    termination = SourceMatchTermination(["ResponderAgent"]) | MaxMessageTermination(8)
    return RoundRobinGroupChat([retriever, responder], termination_condition=termination)


async def run_query(team):
    print(f"user: {USER_QUERY}")
    result = None
    prompt_tokens = completion_tokens = 0
    async for item in team.run_stream(task=USER_QUERY):
        if isinstance(item, TaskResult):
            result = item
            continue
        usage = getattr(item, "models_usage", None)
        if usage:
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
        if isinstance(item, ToolCallRequestEvent):
            for call in item.content:
                print(f"\n{item.source} -> tool call: {call.name}({call.arguments})")
        elif isinstance(item, ToolCallExecutionEvent):
            for output in item.content:
                print(f"{item.source} <- tool result:\n{output.content}")
        elif item.source != "user":
            print(f"\n{item.source}:\n{item.to_text()}")
    print(f"\n(chat tokens: prompt {prompt_tokens}, completion {completion_tokens})")
    return result


async def main():
    print_section("AUTOGEN MULTI-AGENT RAG")
    load_environment()
    model_client, embed = create_clients()
    try:
        team = create_team(model_client, make_retrieve_tips(embed))
        print_section("Query")
        result = await run_query(team)
    finally:
        await model_client.close()

    print_section("Result")
    answered = result is not None and result.messages and result.messages[-1].source == "ResponderAgent"
    print(f"Stop reason: {result.stop_reason if result else 'no result'}")
    if not answered:
        print("FINISHED WITH ERRORS: the ResponderAgent did not answer.")
        sys.exit(1)
    print("COMPLETED: the ResponderAgent answered.")


if __name__ == "__main__":
    http_log.enable()
    asyncio.run(main())
