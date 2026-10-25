"""AutoGen personalized heart-rate lab (3-autogen-personalized-heart-rate.ipynb) as a reduced script, on AutoGen 0.7.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/3-autogen-personalized-heart-rate.ipynb

This script turns a day of sample smartwatch data into an hourly summary and a chart, then has two AutoGen agents plan
the user's day from it. It shows agents working with real data, with the number crunching done in plain code (pandas)
and the planning done by the model. It is a wellness scheduling demo, not medical advice: no health scoring or medical
interpretation.

What it does:
1. Loads the sample wearable data (example-data/20250417_health_data.csv, a heart-rate reading every ~3 seconds),
   summarizes the heart rate per hour and saves a chart of the hourly averages.
2. Runs two agents in a RoundRobinGroupChat: PersonalizedRetrieverAgent calls a local tool that returns the
   hourly summary plus matching wellness tips; PersonalizedResponderAgent writes a schedule of workouts and breaks
   for the user's day, told to plan only from them.

Changes from the notebook:
- Reduced on purpose: the notebook's "heart strain likelihood" score (Low/Moderate/High), strain index, phase
  analysis and mood-based food suggestions are left out.
- The notebook only defined its agents; here they actually run.
- Same connection and fixes as 2-autogen-multi-agent-rag.py:
  - Azure OpenAI deployments of the Foundry project with an Entra ID token (not GitHub Models),
  - tips found by meaning (embeddings, top 3) and passed on word for word,
  - the responder limited to those tips,
  - stop after the responder's answer, exit 1 if it never answers.

Run from the project root:
    uv run python src/4-frameworks/3-autogen-personalized-heart-rate.py [--output-dir DIR]
"""

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the chart to a file; no screen needed
import matplotlib.pyplot as plt
import pandas as pd
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
DATA_FILE = Path(__file__).resolve().parent / "example-data" / "20250417_health_data.csv"
TIMEZONE_OFFSET_SECONDS = 5 * 60 * 60  # same shift to local time as the notebook
USER_QUERY = "Plan my day: when is a good time for a workout, and when should I take a calm break?"
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
    {
        "id": "tip6",
        "content": "For low-intensity recovery days, try yoga or gentle swimming.",
        "source": "Recovery Specialist",
    },
    {
        "id": "tip7",
        "content": "If your heart rate was elevated yesterday, focus on slow, deep breathing exercises today.",
        "source": "Breathing Coach",
    },
    {
        "id": "tip8",
        "content": "Morning walks are ideal when your body is naturally in a recovery state.",
        "source": "Circadian Expert",
    },
    {
        "id": "tip9",
        "content": "Adjust your hydration based on your previous day's heart rate - higher rates mean more water today.",
        "source": "Hydration Specialist",
    },
    {
        "id": "tip10",
        "content": "Schedule high-intensity workouts during your body's natural energy peaks based on heart rate patterns.",
        "source": "Performance Coach",
    },
]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def load_hourly_heart_rate():
    readings = pd.read_csv(DATA_FILE)
    readings["time"] = pd.to_datetime(readings["timestamp"] - TIMEZONE_OFFSET_SECONDS, unit="s")
    hourly = (
        readings.set_index("time")["heart_rate"]
        .resample("h")
        .agg(["count", "mean"])
        .rename(columns={"count": "readings", "mean": "avg_bpm"})
    )
    hourly = hourly[hourly["readings"] >= 300]  # drop hours with only a few minutes of data
    print(
        f"Loaded {len(readings)} readings from {DATA_FILE.name} "
        f"({readings['time'].min():%Y-%m-%d %H:%M} to {readings['time'].max():%H:%M})"
    )
    return hourly


def hourly_summary_text(hourly):
    lowest = hourly["avg_bpm"].nsmallest(3).index
    highest = hourly["avg_bpm"].nlargest(3).index
    lines = [f"{t:%H}:00  average {row.avg_bpm:.0f} bpm" for t, row in hourly.iterrows()]
    return (
        f"Hourly average heart rate yesterday (wearable data, {len(hourly)} hours):\n" + "\n".join(lines) + "\n"
        f"Overall average: {hourly['avg_bpm'].mean():.0f} bpm\n"
        f"Calmest hours (lowest averages): {', '.join(f'{t:%H}:00' for t in sorted(lowest))}\n"
        f"Most active hours (highest averages): {', '.join(f'{t:%H}:00' for t in sorted(highest))}"
    )


def save_chart(hourly, output_dir):
    path = Path(output_dir) / "hourly-heart-rate.png"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hourly.index, hourly["avg_bpm"], marker="o")
    ax.set_title("Hourly average heart rate (sample wearable data)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average heart rate (bpm)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:00"))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved chart: {path}")


def make_retrieve_personalized_tips(summary, embed):
    """Embed the tips once; the returned tool embeds each query and returns the closest tips with the summary."""
    tip_vectors = embed([tip["content"] for tip in HEALTH_TIPS])
    print(f"Embedded {len(HEALTH_TIPS)} wellness tips.")

    def retrieve_personalized_tips(query: str) -> str:
        """Return the user's hourly heart-rate summary from yesterday plus the wellness tips closest in meaning to the question."""
        query_vector = embed([query])[0]
        ranked = sorted(
            zip(HEALTH_TIPS, (cosine(query_vector, v) for v in tip_vectors), strict=True),
            key=lambda p: p[1],
            reverse=True,
        )
        print(
            f"  [tool] retrieve_personalized_tips(query={query!r}) top similarity: "
            + ", ".join(f"{tip['id']} {score:.2f}" for tip, score in ranked[:5])
        )
        tip_text = "\n".join(f"Source: {tip['source']} => {tip['content']}" for tip, _ in ranked[:TOP_TIPS])
        return f"YOUR HEART RATE DATA:\n{summary}\n\nWELLNESS TIPS:\n{tip_text}"

    return retrieve_personalized_tips


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


def create_team(model_client, retrieve_personalized_tips):
    retriever = AssistantAgent(
        name="PersonalizedRetrieverAgent",
        system_message=(
            "You are a retrieval agent. Call the tool 'retrieve_personalized_tips' once with the user's question."
        ),
        model_client=model_client,
        tools=[retrieve_personalized_tips],
        reflect_on_tool_use=False,  # pass the summary and tips on word for word
    )
    responder = AssistantAgent(
        name="PersonalizedResponderAgent",
        system_message=(
            "You are a friendly wellness coach. Plan the user's day using only the hourly heart-rate summary and the "
            "wellness tips provided: suggest workouts in the most active hours and calm breaks in the calmest hours, "
            "and name the source of each tip you use. Do not add tips or advice that are not in the list. This is a "
            "wellness demo: do not assess health risks, diagnose, or interpret heart rate as a medical sign, and say "
            "it is not medical advice."
        ),
        model_client=model_client,
    )
    termination = SourceMatchTermination(["PersonalizedResponderAgent"]) | MaxMessageTermination(8)
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
    parser = argparse.ArgumentParser(description="AutoGen personalized heart-rate lab (reduced)")
    parser.add_argument("--output-dir", default=".", help="folder for the chart (default: current folder)")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print_section("AUTOGEN PERSONALIZED HEART-RATE AGENTS (REDUCED)")
    print("Wellness demo only: no health scoring and no medical advice.")
    load_environment()

    print_section("1. Hourly heart-rate summary")
    hourly = load_hourly_heart_rate()
    summary = hourly_summary_text(hourly)
    print(summary)
    save_chart(hourly, args.output_dir)

    print_section("2. Agents")
    model_client, embed = create_clients()
    try:
        result = await run_query(create_team(model_client, make_retrieve_personalized_tips(summary, embed)))
    finally:
        await model_client.close()

    print_section("Result")
    answered = result is not None and result.messages and result.messages[-1].source == "PersonalizedResponderAgent"
    print(f"Stop reason: {result.stop_reason if result else 'no result'}")
    if not answered:
        print("FINISHED WITH ERRORS: the PersonalizedResponderAgent did not answer.")
        sys.exit(1)
    print("COMPLETED: the PersonalizedResponderAgent answered.")


if __name__ == "__main__":
    http_log.enable()
    asyncio.run(main())
