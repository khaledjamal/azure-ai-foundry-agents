"""Foundry agent (new agents API, azure-ai-projects 2.x) with the built-in Web Search
tool, next to the Bing lab.

New: not in Microsoft's workshop.

This script is a companion to the Bing lab (4-bing-grounding.py): the same three health
questions, instructions and checks, but the agent uses the newer built-in Web Search
tool (`WebSearchTool` instead of `BingGroundingTool`).

Why this tool:
- Microsoft's docs now say to start with it: it needs no Grounding with Bing resource
  and no project connection, so the agent definition holds only the tool.
- Underneath it still uses Grounding with Bing, so Bing's terms apply and the queries
  leave the Azure compliance boundary, as in the Bing lab.
- In the recorded run it cited more sources (11 against 7) and read about twice the
  input tokens, still under a cent: no quality gap, and less to set up, which makes it
  the simpler default for a new agent.

What it does:
1. Asks each question in its own response with tool_choice="required".
2. For each answer, prints the text with numbered citations, the source websites and the
   query of each search (from the `web_search_call` items), with a Bing search URL built
   from it.
3. Deletes the agent version at the end.
Exits with 1 unless every question got an answer with at least one source.

Settings from .env: AZURE_AI_SERVICES_ENDPOINT (or AZURE_AI_PROJECT_ENDPOINT),
AZURE_AI_PROJECT_NAME, MODEL_DEPLOYMENT_NAME. Paid: the Agent Service lists Web Search
per 1,000 transactions without a published rate; Grounding with Bing is $14 per 1,000
(checked 2026-10-18), so a run with 3-9 searches is about $0.04-0.13, plus gpt-4o-mini
tokens.

The instructions carry today's date, as in the Bing lab. --today sets that date
(default: today), so a recorded run can be replayed on a later day with the recording's
date.

Needs azure-ai-projects>=2 (in the separate agents venv). Run from the project root:
    source ../../.venv-agents-v2/bin/activate
    python src/2-agents/4-web-search.py [--model gpt-4o-mini] [--today YYYY-MM-DD]
"""

import argparse
import os
from datetime import date
from urllib.parse import quote_plus

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, WebSearchTool
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

AGENT_NAME = "health-web-search-agent"
# The same questions as the Bing lab (4-bing-grounding.py), so the two runs can be
# compared.
QUESTIONS = [
    "What are some new HIIT workout trends I should know about?",
    "What's the current WHO recommendation for sugar intake?",
    "Any news on intermittent fasting for weight management?",
]
PRICE_PER_SEARCH = 14 / 1000  # USD, Grounding with Bing rate, checked 2026-10-18


def build_instructions(today):
    return (
        f"Today's date is {today:%B %d, %Y}. You are a health and fitness assistant with web search. Search the web "
        "for every question; for questions about news, trends or current guidance, search for results that are "
        "recent as of today's date. Prefer official health sources (for example WHO, NIH, CDC, NHS or a national "
        "health agency) and say so when you found only commercial or blog sources. Answer only from what the search "
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
    """Replace each cited span with [n] where it is a raw marker; return the new text
    and the numbered sources."""
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
    """Print one response's output items; return (answer text, sources, search
    queries)."""
    answer, sources, queries = "", [], []
    for item in response.output:
        if item.type == "web_search_call":
            action = getattr(item, "action", None)
            query = getattr(action, "query", None) or ""
            if query:
                queries.append(query)
            print(f"  web_search_call  action={getattr(action, 'type', None)}  query={query!r}  status={item.status}")
        elif item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    answer, sources = number_citations(content.text, content.annotations or [])
            print("  message (answer, printed below)")
        else:
            print(f"  {item.type}  status={getattr(item, 'status', None)}")
    if response.usage:
        print(f"  (tokens: input {response.usage.input_tokens}, output {response.usage.output_tokens})")
    return answer, sources, queries


def parse_args():
    parser = argparse.ArgumentParser(description="Foundry agent with the built-in Web Search tool")
    parser.add_argument("--model", default=os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini")
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="date written into the instructions (default: today); a replay passes the recording's date",
    )
    return parser.parse_args()


def load_settings(model):
    project_endpoint = get_project_endpoint()
    print(f"Project endpoint: {project_endpoint}")
    print(f"Model deployment: {model}")
    return project_endpoint


def create_agent(project, model, today):
    """Create an agent version with the built-in Web Search tool; it needs no project
    connection."""
    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(model=model, instructions=build_instructions(today), tools=[WebSearchTool()]),
    )
    print(f"\nAgent created: name={agent.name} version={agent.version} (tool: web_search, no connection)")
    return agent


def ask(openai_client, agent_reference, question, totals):
    """Send one question with a required search; return the response status, answer,
    sources and search queries."""
    print(f"\nUser: {question}")
    response = openai_client.responses.create(input=question, tool_choice="required", extra_body=agent_reference)
    print(f"Response (status {response.status}):")
    answer, sources, queries = show_response(response)
    if response.usage:
        totals["input"] += response.usage.input_tokens
        totals["output"] += response.usage.output_tokens
    totals["searches"] += sum(1 for item in response.output if item.type == "web_search_call")
    return response.status, answer, sources, queries


def print_answer(answer, sources, queries):
    print(f"\nAgent: {answer}")
    print("Sources:")
    for number, (title, url) in enumerate(sources, start=1):
        print(f"  [{number}] {title}: {url}")
    if not sources:
        print("  (none)")
    for query in queries:
        print(f"Bing search: https://www.bing.com/search?q={quote_plus(query)}")


def ask_questions(openai_client, agent):
    """Ask each question in its own response; return (question, status, answer, sources)
    and the totals."""
    agent_reference = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
    totals = {"input": 0, "output": 0, "searches": 0}
    results = []
    for question in QUESTIONS:
        status, answer, sources, queries = ask(openai_client, agent_reference, question, totals)
        print_answer(answer, sources, queries)
        results.append((question, status, answer, sources))
    return results, totals


def cleanup(project, agent):
    project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
    print(f"\nAgent version deleted: {agent.name} v{agent.version}")


def print_totals(totals):
    print(f"Total tokens: input {totals['input']}, output {totals['output']}")
    print(f"Web searches: {totals['searches']} (about ${totals['searches'] * PRICE_PER_SEARCH:.3f} at the Bing rate)")


def check_results(results):
    """Every question got a completed answer with at least one source."""
    checks = [
        (f"answered with sources: {question}", status == "completed" and bool(answer.strip()) and bool(sources))
        for question, status, answer, sources in results
    ]
    print("Checks:")
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return len(checks) == len(QUESTIONS) and all(passed for _, passed in checks)


def main():
    load_env()
    args = parse_args()
    project_endpoint = load_settings(args.model)

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project:
        openai_client = project.get_openai_client()
        agent = create_agent(project, args.model, args.today)
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
