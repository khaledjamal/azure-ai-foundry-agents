"""Foundry agent (new agents API, azure-ai-projects 2.x) with an Azure AI Search tool
and a local price function tool.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/5-agents-aisearch.ipynb

This script moves the shopping assistant of 5-agents-aisearch.py from the classic agents
API, which Microsoft retires on 2027-03-31, to the new one, and compares models on it.
Newer models such as gpt-5.4-mini fail with the search tool on the classic API, so this
is where they can be compared with gpt-4o-mini.

Same as 5-agents-aisearch.py:
- The small fitness catalog, the two questions (each model answers both), the
  instructions and the price logic.
- The same two tools: Azure AI Search, to find products, and a price filter
  (find_items_under) that runs in this script, to answer budget questions exactly.
- The price is also in each product's text, because the search tool passes only text
  fields to the model.

Differences on the new API:
- One versioned prompt agent per model (agents.create_version), called through the
  Responses API with an agent reference; no threads or runs.
- The function tool is declared as a JSON schema. The script runs the price filter
  itself: when a response contains function_call items, it runs find_items_under and
  sends function_call_output items back with previous_response_id, until the model
  answers.
- The search tool reaches the index through the project's Azure AI Search connection
  (project_connection_id).

Checks per model (exits with 1 if any fails):
- both questions answered,
- question 1 used the search tool,
- question 2 called find_items_under(max_price=300).
A table at the end shows the tool calls, tokens (including the model's thinking), time
and cost per model. Each agent version is deleted after its questions, the index at the
end.

Settings from .env: AZURE_AI_SERVICES_ENDPOINT (or AZURE_AI_PROJECT_ENDPOINT),
AZURE_AI_PROJECT_NAME, MODEL_DEPLOYMENT_NAME (default model), AZURE_AI_SEARCH_ENDPOINT,
AZURE_AI_SEARCH_API_KEY, optional AZURE_AI_SEARCH_CONNECTION_NAME. Paid: under $0.01 per
run on gpt-4o-mini; prices per model are in PRICES below.

Needs azure-ai-projects>=2 and azure-search-documents (in the separate agents venv). Run
from the project root:
    source ../../.venv-agents-v2/bin/activate
    python src/2-agents/5-agents-aisearch-v2.py
        [--models gpt-4o-mini,gpt-5.4-mini,gpt-5.4-nano]
"""

import argparse
import json
import os
import time

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AISearchIndexResource,
    AzureAISearchQueryType,
    AzureAISearchTool,
    AzureAISearchToolResource,
    ConnectionType,
    FunctionTool,
    PromptAgentDefinition,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchableField, SearchFieldDataType, SearchIndex, SimpleField

from workshop_utils import http_log
from workshop_utils.settings import load_env

INDEX_NAME = "myfitnessindex-v2"
AGENT_NAME = "fitness-agent-search-v2"
MAX_FUNCTION_ROUNDS = 4
OUTPUT_EXCERPT = 400

# USD per 1M input / output tokens, Global deployment in East US 2, Azure Retail Prices
# API, checked 2026-10-23. Reasoning tokens are billed as output tokens (and are
# included in the output count).
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
}

QUESTIONS = [
    "Which items are good for strength training?",
    "I need something for cardio under $300, any suggestions?",
]

SAMPLE_DOCUMENTS = [
    {
        "FitnessItemID": "1",
        "Name": "Adjustable Dumbbell",
        "Category": "Strength",
        "Price": 59.99,
        "Description": "A compact, adjustable weight for targeted muscle workouts.",
    },
    {
        "FitnessItemID": "2",
        "Name": "Yoga Mat",
        "Category": "Flexibility",
        "Price": 25.0,
        "Description": "Non-slip mat designed for yoga, Pilates, and other exercises.",
    },
    {
        "FitnessItemID": "3",
        "Name": "Treadmill",
        "Category": "Cardio",
        "Price": 499.0,
        "Description": "A sturdy treadmill with adjustable speed and incline settings.",
    },
    {
        "FitnessItemID": "4",
        "Name": "Resistance Bands",
        "Category": "Strength",
        "Price": 15.0,
        "Description": "Set of colorful bands for light to moderate resistance workouts.",
    },
]

AGENT_INSTRUCTIONS = """
You are a Fitness Shopping Assistant. You help users find fitness equipment in the store's catalog.

Always look products up with the Azure AI Search tool before answering.
When the user gives a budget or price limit, call find_items_under to get the exact products within that price;
do not compare prices yourself.
Recommend only products returned by the search or by find_items_under, and state the price of each one.
Do not invent products, prices, categories, or product details that are not in the search index.
If nothing matches, say so clearly and mention the closest option.
Any health or fitness information you provide is for general educational purposes only and is not medical advice.
""".strip()

FIND_ITEMS_UNDER_TOOL = FunctionTool(
    name="find_items_under",
    description=(
        "Find fitness products priced at or below max_price, optionally in one category (Strength, Flexibility or "
        "Cardio); use it whenever the user gives a budget or price limit."
    ),
    parameters={
        "type": "object",
        "properties": {
            "max_price": {"type": "number", "description": "Maximum price in US dollars."},
            "category": {
                "type": ["string", "null"],
                "description": "Product category (Strength, Flexibility or Cardio), or null for all categories.",
            },
        },
        "required": ["max_price", "category"],
        "additionalProperties": False,
    },
    strict=True,
)


def first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_project_endpoint():
    direct_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if direct_endpoint:
        return direct_endpoint.rstrip("/")
    services_endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/")
    project_name = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{services_endpoint}/api/projects/{project_name}"


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def create_search_clients():
    search_endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
    credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_API_KEY"])
    print(f"Azure AI Search endpoint: {search_endpoint}")
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    search_client = SearchClient(endpoint=search_endpoint, index_name=INDEX_NAME, credential=credential)
    return index_client, search_client


def create_fitness_index(index_client, search_client):
    fields = [
        SimpleField(name="FitnessItemID", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="Name", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="Category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="Price", type=SearchFieldDataType.Double, filterable=True, sortable=True, facetable=True),
        SearchableField(name="Description", type=SearchFieldDataType.String),
    ]
    if INDEX_NAME in {index.name for index in index_client.list_indexes()}:
        index_client.delete_index(INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")
        time.sleep(1)
    index_client.create_index(SearchIndex(name=INDEX_NAME, fields=fields))
    print(f"Created index: {INDEX_NAME}")

    # The search tool passes only text fields to the model, so the price also goes into
    # the description.
    documents = [{**d, "Description": f"{d['Description']} Price: ${d['Price']:.2f}."} for d in SAMPLE_DOCUMENTS]
    failed = [result.key for result in search_client.upload_documents(documents=documents) if not result.succeeded]
    if failed:
        raise RuntimeError(f"Documents failed to upload: {failed}")
    print(f"Uploaded {len(documents)} products.")

    # A new index can take a moment before its documents are searchable; wait instead of
    # letting the agent search an empty index.
    for _ in range(10):
        if len(list(search_client.search(search_text="*", top=10))) == len(documents):
            print("Index is searchable.")
            return
        time.sleep(2)
    raise RuntimeError("The index did not return all products after 20 seconds.")


def make_find_items_under(search_client):
    # Same logic as find_items_under in 5-agents-aisearch.py and the Semantic Kernel
    # plugin.
    def products(filter_expression, top=10):
        results = search_client.search(
            search_text="*",
            filter=filter_expression,
            order_by=["Price asc"],
            select=["Name", "Category", "Price"],
            top=top,
        )
        return [f"{d['Name']} ({d['Category']}): ${d['Price']:.2f}" for d in results]

    def find_items_under(max_price, category=None):
        category = category.strip().title() if category and category.strip() else None
        category_filter = f" and Category eq '{category.replace(chr(39), chr(39) * 2)}'" if category else ""
        matches = products(f"Price le {max_price}{category_filter}")
        if matches:
            return "\n".join(matches)
        answer = f"No products cost ${max_price:.2f} or less" + (f" in category {category}." if category else ".")
        if category:
            cheapest = products(category_filter.removeprefix(" and "), top=1)
            if cheapest:
                answer += f" Cheapest product in {category}: {cheapest[0]}"
        return answer

    return find_items_under


def find_search_connection(project):
    name = first_env("AZURE_AI_SEARCH_CONNECTION_NAME", "AI_SEARCH_CONNECTION_NAME")
    connection = (
        project.connections.get(name) if name else project.connections.get_default(ConnectionType.AZURE_AI_SEARCH)
    )
    print(f"Azure AI Search connection: {connection.name}")
    return connection


def excerpt(value):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= OUTPUT_EXCERPT else text[:OUTPUT_EXCERPT] + " ..."


def show_output(response, stats):
    """Print one response's output items and add its tool calls and token usage to
    stats."""
    for item in response.output:
        if item.type == "function_call":
            print(f"  function_call  {item.name}({item.arguments})")
            stats["functions"].append((item.name, json.loads(item.arguments or "{}")))
        elif item.type == "message":
            print("  message (answer, printed below)")
        elif item.type == "reasoning":
            print("  reasoning (the model's thinking; its tokens are counted below)")
        else:
            # azure_ai_search_call is the search; its result comes back as a separate
            # azure_ai_search_call_output.
            if item.type.endswith("search_call"):
                stats["searches"] += 1
            print(f"  {item.type}  status={getattr(item, 'status', None)}")
            for field in ("arguments", "output", "error"):
                value = getattr(item, field, None)
                if value:
                    print(f"    {field}: {excerpt(value)}")
    usage = response.usage
    if usage:
        details = getattr(usage, "output_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        stats["input"] += usage.input_tokens
        stats["output"] += usage.output_tokens
        stats["reasoning"] += reasoning
        print(f"  (tokens: input {usage.input_tokens}, output {usage.output_tokens}, of which reasoning {reasoning})")


def ask(openai_client, agent_reference, question, find_items_under):
    """One question: send it, run every function call the model asks for, and return the
    final answer and stats."""
    stats = {"searches": 0, "functions": [], "input": 0, "output": 0, "reasoning": 0}
    print(f"\nUser: {question}")
    started = time.monotonic()
    response = openai_client.responses.create(input=question, extra_body=agent_reference)
    print(f"Response (status {response.status}):")
    show_output(response, stats)
    for _ in range(MAX_FUNCTION_ROUNDS):
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break
        outputs = []
        for call in calls:
            if call.name == "find_items_under":
                output = find_items_under(**json.loads(call.arguments or "{}"))
            else:
                output = f"Unknown function: {call.name}"
            print(f"  -> ran {call.name} here: {output!r}")
            outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
        response = openai_client.responses.create(
            input=outputs, previous_response_id=response.id, extra_body=agent_reference
        )
        print(f"Response (status {response.status}):")
        show_output(response, stats)
    stats["seconds"] = time.monotonic() - started
    stats["status"] = response.status
    stats["answer"] = response.output_text or ""
    print(f"\nAgent: {stats['answer']}")
    return stats


def run_model(project, openai_client, model, search_tool, find_items_under):
    """Create an agent version on this model, ask every question, delete the version;
    return one stats per question."""
    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=model, instructions=AGENT_INSTRUCTIONS, tools=[search_tool, FIND_ITEMS_UNDER_TOOL]
        ),
    )
    print(f"Agent created: name={agent.name} version={agent.version} model={model}")
    agent_reference = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
    try:
        return [ask(openai_client, agent_reference, question, find_items_under) for question in QUESTIONS]
    finally:
        project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        print(f"\nAgent version deleted: {agent.name} v{agent.version}")


def model_checks(results):
    if len(results) != len(QUESTIONS):
        return [("all questions asked", False)]
    question_1, question_2 = results
    return [
        ("both questions answered", all(r["status"] == "completed" and r["answer"].strip() for r in results)),
        ("question 1 used the search tool", question_1["searches"] > 0),
        (
            "question 2 called find_items_under(max_price=300)",
            any(name == "find_items_under" and args.get("max_price") == 300 for name, args in question_2["functions"]),
        ),
    ]


def cost(model, input_tokens, output_tokens):
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def print_summary(all_results):
    print_section("COMPARISON (both questions per model)")
    header = (
        f"{'model':<14} {'searches':>8} {'functions':>9} {'input':>7} {'output':>7} {'reasoning':>9} "
        f"{'time':>7} {'cost':>9}"
    )
    print(header)
    print("-" * len(header))
    for model, results in all_results.items():
        totals = {key: sum(r[key] for r in results) for key in ("searches", "input", "output", "reasoning", "seconds")}
        functions = sum(len(r["functions"]) for r in results)
        price = cost(model, totals["input"], totals["output"])
        price_text = f"${price:.4f}" if price is not None else "no price"
        print(
            f"{model:<14} {totals['searches']:>8} {functions:>9} {totals['input']:>7} {totals['output']:>7} "
            f"{totals['reasoning']:>9} {totals['seconds']:>6.1f}s {price_text:>9}"
        )
    print("Prices: list prices per 1M tokens (see PRICES), Global deployment in East US 2, checked 2026-10-23.")


def model_list(value):
    """--models value: comma-separated names, blanks dropped."""
    return [model.strip() for model in value.split(",") if model.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Foundry agent (new agents API) with Azure AI Search and a price tool")
    parser.add_argument(
        "--models",
        type=model_list,
        default=os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini",
        help="comma-separated model deployment names; each gets its own agent version and the same questions",
    )
    return parser.parse_args()


def load_settings(models):
    project_endpoint = get_project_endpoint()
    print(f"Project endpoint: {project_endpoint}")
    print(f"Models: {', '.join(models)}")
    print("Disclaimer: This program is a technical demonstration and does not provide professional medical advice.")
    return project_endpoint


def make_search_tool(connection):
    """The Azure AI Search tool on the index, reached through the project's Azure AI
    Search connection."""
    return AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=connection.id,
                    index_name=INDEX_NAME,
                    query_type=AzureAISearchQueryType.SIMPLE,
                    top_k=5,
                )
            ]
        )
    )


def run_models(project, models, search_tool, find_items_under):
    """Step 2: run the questions on an agent version per model; return {model: stats per
    question}."""
    openai_client = project.get_openai_client()
    all_results = {}
    for model in models:
        print_section(f"STEP 2: AGENT ON {model}")
        # One model failing (for example a deployment that doesn't exist) shouldn't hide
        # the others' results.
        try:
            all_results[model] = run_model(project, openai_client, model, search_tool, find_items_under)
        except Exception as exc:
            print(f"Model {model} failed: {exc!r}")
            all_results[model] = []
    return all_results


def cleanup(index_client):
    try:
        if INDEX_NAME in {index.name for index in index_client.list_indexes()}:
            index_client.delete_index(INDEX_NAME)
            print(f"Deleted index: {INDEX_NAME}")
    except Exception as exc:
        print(f"Could not delete index {INDEX_NAME}: {exc}")


def check_results(all_results, models):
    """Every model must have run and passed model_checks()."""
    print("Checks:")
    passed_all = len(all_results) == len(models)
    for model, results in all_results.items():
        for name, passed in model_checks(results):
            print(f"  {'PASS' if passed else 'FAIL'}  {model}: {name}")
            passed_all = passed_all and passed
    return passed_all


def main():
    load_env()
    args = parse_args()
    project_endpoint = load_settings(args.models)

    print_section("STEP 1: CREATE AND FILL THE AZURE AI SEARCH INDEX")
    index_client, search_client = create_search_clients()
    all_results = {}
    try:
        create_fitness_index(index_client, search_client)
        find_items_under = make_find_items_under(search_client)

        with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project:
            connection = find_search_connection(project)
            search_tool = make_search_tool(connection)
            all_results = run_models(project, args.models, search_tool, find_items_under)
    finally:
        print_section("CLEANUP")
        cleanup(index_client)

    print_summary(all_results)
    passed = check_results(all_results, args.models)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
