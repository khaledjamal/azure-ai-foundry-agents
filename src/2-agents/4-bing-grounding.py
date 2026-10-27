"""Foundry agent (new agents API, azure-ai-projects 2.x) that grounds its answers with Grounding with Bing Search.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/4-bing_grounding.ipynb

This script builds an agent that searches the web with Grounding with Bing before it answers, for questions the model's
training data can't cover, such as recent health news. It is the script version of 4-bing_grounding.ipynb, written fresh
for the new agents API (it replaces an earlier classic-API version, old-4-bing_grounding.py).

What it does:
1. Creates the agent. Its Bing tool uses the project connection named in GROUNDING_WITH_BING_CONNECTION_NAME, so the
   Bing key stays in a Foundry project connection and the agent definition holds no key.
2. Asks the notebook's three health questions, each in its own response with tool_choice="required", so every
   question runs at least one Bing search.
3. For each answer, prints the text with numbered citations, the source websites and the Bing search URL of each
   search (built from the query the agent sent), because Grounding with Bing's use and display requirements ask for
   both.
4. Deletes the agent version at the end.
Exits with 1 unless every question got an answer with at least one source and a Bing search URL.

Finding: the model writes the Bing query itself and otherwise dates "news" by its training data (one run searched
"... news 2023"). The fix: the instructions carry today's date and ask for official health sources first. --today
sets that date (default: today), so a recorded run can be replayed on a later day with the recording's date.

Settings from .env: GROUNDING_WITH_BING_CONNECTION_NAME, AZURE_AI_SERVICES_ENDPOINT (or
AZURE_AI_PROJECT_ENDPOINT), AZURE_AI_PROJECT_NAME, MODEL_DEPLOYMENT_NAME. Paid: Grounding with Bing costs $14 per
1,000 searches (checked 2026-10-18), so a run with 3-9 searches is about $0.04-0.13, plus gpt-4o-mini tokens.

Needs azure-ai-projects>=2 (in the separate agents venv). Run from the project root:
    source ../../.venv-agents-v2/bin/activate
    python src/2-agents/4-bing-grounding.py [--model gpt-4o-mini] [--today YYYY-MM-DD]
"""

import argparse
import json
import os
from datetime import date
from urllib.parse import quote_plus

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    BingGroundingSearchConfiguration,
    BingGroundingSearchToolParameters,
    BingGroundingTool,
    PromptAgentDefinition,
)
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

AGENT_NAME = "health-bing-agent"
QUESTIONS = [
    "What are some new HIIT workout trends I should know about?",
    "What's the current WHO recommendation for sugar intake?",
    "Any news on intermittent fasting for weight management?",
]
BING_PRICE_PER_SEARCH = 14 / 1000  # USD, Grounding with Bing Search, checked 2026-10-18


def build_instructions(today):
    return (
        f"Today's date is {today:%B %d, %Y}. You are a health and fitness assistant with Bing search. Search Bing for "
        "every question; for questions about news, trends or current guidance, search for results that are recent "
        "as of today's date. Prefer official health sources (for example WHO, NIH, CDC, NHS or a national health "
        "agency) and say so when you found only commercial or blog sources. Answer only from what the search "
        "returns, citing your sources. Keep answers brief and helpful. Say that you are not a medical professional "
        "and encourage the user to consult one."
    )


def get_project_endpoint():
    direct_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if direct_endpoint:
        return direct_endpoint.rstrip("/")
    services_endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/")
    project_name = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{services_endpoint}/api/projects/{project_name}"


def number_citations(text, annotations):
    """Replace each cited marker (e.g. 【4:0†source】) with [n]; return the new text and the numbered (title, url) list."""
    sources, replacements = [], []
    for annotation in annotations:
        if annotation.type != "url_citation":
            continue
        source = (annotation.title, annotation.url)
        if source not in sources:
            sources.append(source)
        start, end = getattr(annotation, "start_index", None), getattr(annotation, "end_index", None)
        if start is not None and end is not None and text[start:end].startswith("【"):
            replacements.append((start, end, f"[{sources.index(source) + 1}]"))
    for start, end, label in sorted(replacements, reverse=True):
        text = text[:start] + label + text[end:]
    return text, sources


def show_response(response):
    """Print one response's output items; return (answer text, sources, Bing search URLs)."""
    answer, sources, search_urls = "", [], []
    for item in response.output:
        if item.type == "bing_grounding_call":
            query = json.loads(item.arguments or "{}").get("query", "")
            search_urls.append(f"https://www.bing.com/search?q={quote_plus(query)}")
            print(f"  bing_grounding_call  query={query!r}  status={item.status}")
        elif item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    answer, sources = number_citations(content.text, content.annotations or [])
            print("  message (answer, printed below)")
        else:
            print(f"  {item.type}  status={getattr(item, 'status', None)}")
    if response.usage:
        print(f"  (tokens: input {response.usage.input_tokens}, output {response.usage.output_tokens})")
    return answer, sources, search_urls


def parse_args():
    parser = argparse.ArgumentParser(description="Foundry agent grounded with Grounding with Bing Search")
    parser.add_argument("--model", default=os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini")
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="date written into the instructions (default: today); a replay passes the recording's date",
    )
    return parser.parse_args()


def load_settings(model):
    connection_name = os.environ["GROUNDING_WITH_BING_CONNECTION_NAME"]
    project_endpoint = get_project_endpoint()
    print(f"Project endpoint: {project_endpoint}")
    print(f"Model deployment: {model}")
    return project_endpoint, connection_name


def create_agent(project, model, connection_name, today):
    """Create an agent version with the Bing tool; the Bing key stays in the project connection."""
    connection = project.connections.get(connection_name)
    print(f"Bing connection: {connection.name} ({connection.type})")
    tool = BingGroundingTool(
        bing_grounding=BingGroundingSearchToolParameters(
            search_configurations=[BingGroundingSearchConfiguration(project_connection_id=connection.id)]
        )
    )
    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(model=model, instructions=build_instructions(today), tools=[tool]),
    )
    print(f"\nAgent created: name={agent.name} version={agent.version}")
    return agent


def ask(openai_client, agent_reference, question, totals):
    """Send one question with a required search; return the response status, answer, sources and search URLs."""
    print(f"\nUser: {question}")
    response = openai_client.responses.create(input=question, tool_choice="required", extra_body=agent_reference)
    print(f"Response (status {response.status}):")
    answer, sources, search_urls = show_response(response)
    if response.usage:
        totals["input"] += response.usage.input_tokens
        totals["output"] += response.usage.output_tokens
    totals["searches"] += len(search_urls)
    return response.status, answer, sources, search_urls


def print_answer(answer, sources, search_urls):
    print(f"\nAgent: {answer}")
    print("Sources:")
    for number, (title, url) in enumerate(sources, start=1):
        print(f"  [{number}] {title}: {url}")
    if not sources:
        print("  (none)")
    for url in search_urls:
        print(f"Bing search: {url}")


def ask_questions(openai_client, agent):
    """Ask each question in its own response; return (question, status, answer, sources, search URLs) and the totals."""
    agent_reference = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
    totals = {"input": 0, "output": 0, "searches": 0}
    results = []
    for question in QUESTIONS:
        status, answer, sources, search_urls = ask(openai_client, agent_reference, question, totals)
        print_answer(answer, sources, search_urls)
        results.append((question, status, answer, sources, search_urls))
    return results, totals


def cleanup(project, agent):
    project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
    print(f"\nAgent version deleted: {agent.name} v{agent.version}")


def print_totals(totals):
    print(f"Total tokens: input {totals['input']}, output {totals['output']}")
    print(f"Bing searches: {totals['searches']} (about ${totals['searches'] * BING_PRICE_PER_SEARCH:.3f})")


def check_results(results):
    """Every question got a completed answer with at least one source and a Bing search URL."""
    checks = [
        (
            f"answered with sources and a Bing search URL: {question}",
            status == "completed" and bool(answer.strip()) and bool(sources) and bool(search_urls),
        )
        for question, status, answer, sources, search_urls in results
    ]
    print("Checks:")
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return len(checks) == len(QUESTIONS) and all(passed for _, passed in checks)


def main():
    load_env()
    args = parse_args()
    project_endpoint, connection_name = load_settings(args.model)

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project:
        openai_client = project.get_openai_client()
        agent = create_agent(project, args.model, connection_name, args.today)
        try:
            results, totals = ask_questions(openai_client, agent)
        finally:
            cleanup(project, agent)

    print_totals(totals)
    passed = check_results(results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
