"""Semantic Kernel + Azure AI Search lab (1-rag-sk-agents-aisearch.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/1-rag-sk-agents-aisearch.ipynb

This script builds the same kind of shopping assistant as the Azure AI Search agent lab,
but through Semantic Kernel, Microsoft's open-source framework for AI apps. On gpt-4o it
made the same tool calls and gave the same answers as on gpt-4o-mini, at about 17 times
the cost (about $0.011 against $0.0007 per run).

What it does:
1. Builds a small fitness product index in Azure AI Search.
2. Creates a Foundry agent with the Azure AI Search tool and wraps it in Semantic
   Kernel's AzureAIAgent.
3. Asks two shopping questions on one thread.
4. Deletes everything it created (thread, agent, index).

Changes from the notebook (it used Semantic Kernel calls that no longer exist):
- Current Semantic Kernel API: AzureAIAgent.create_client() with the Foundry project
  endpoint (the notebook's connection string and AzureAIAgentSettings.create() no longer
  exist), AzureAIAgentThread and get_response().
- Price as text: each product's Description also states its price ("Price: $499.00.").
  The numeric Price field is not searchable, so on its own it never reaches the model
  through the search tool. This is the same problem as in the agent lab, fixed the same
  way.
- Price filter plugin: a Semantic Kernel plugin (find_items_under), code running here in
  this script, queries the index with a real filter (Price le <max>), so a budget
  question gets an exact answer instead of the model comparing prices itself.
- Tool output: after each answer the script prints the tools offered to the model in
  that run and every tool call with what it returned, so the output shows which product
  fields the search tool actually passed on.

Connection method:
- Client: AIProjectClient (async) from azure-ai-projects, created by Semantic Kernel
  with the Foundry project endpoint and an Entra ID token from the Azure CLI login.
- Azure AI Search: the endpoint and key come from the project's default Azure AI Search
  connection.

Uses its own environment (see requirements.txt in this folder). Run from the project
root:
    .venv-semantic-kernel/bin/python src/4-frameworks/1-rag-sk-agents-aisearch.py
"""

import asyncio
import os
from typing import Annotated

from azure.ai.agents.models import AzureAISearchQueryType, AzureAISearchTool, ListSortOrder
from azure.ai.projects.models import ConnectionType
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchableField, SearchFieldDataType, SearchIndex, SimpleField
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
from semantic_kernel.functions import kernel_function

from workshop_utils import http_log
from workshop_utils.settings import load_env

INDEX_NAME = "fitness-sk-index"
AGENT_NAME = "fitness-sk-agent"

QUESTIONS = [
    "Which items are best for strength training?",
    "I need something for cardio under $300. Any suggestions?",
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
Do not invent products, prices or details.
If nothing matches, say so clearly and mention the closest option.
Always include a disclaimer that you are not providing medical advice.
""".strip()


class FitnessCatalogPlugin:
    """Local tool for the agent: exact price filtering on the search index (runs in this
    script, not in Azure)."""

    def __init__(self, search_client):
        self._search_client = search_client

    @kernel_function(
        name="find_items_under",
        description=(
            "Find fitness products whose price is at or below a maximum price, optionally in one category "
            "(Strength, Flexibility or Cardio). Use it whenever the user gives a budget or price limit."
        ),
    )
    async def find_items_under(
        self,
        max_price: Annotated[float, "Maximum price in US dollars"],
        category: Annotated[str | None, "Product category, or empty for all categories"] = None,
    ) -> Annotated[str, "Matching products with their prices"]:
        category = category.strip().title() if category and category.strip() else None
        category_filter = f" and Category eq '{category.replace(chr(39), chr(39) * 2)}'" if category else ""

        matches = await self._products(f"Price le {max_price}{category_filter}")
        print(f"  [plugin] find_items_under(max_price={max_price}, category={category}) -> {len(matches)} item(s)")
        if matches:
            return "\n".join(matches)

        answer = f"No products cost ${max_price:.2f} or less" + (f" in category {category}." if category else ".")
        if category:
            cheapest = await self._products(category_filter.removeprefix(" and "), top=1)
            if cheapest:
                answer += f" Cheapest product in {category}: {cheapest[0]}"
        return answer

    async def _products(self, filter_expression, top=10):
        results = await self._search_client.search(
            search_text="*",
            filter=filter_expression,
            order_by=["Price asc"],
            select=["Name", "Category", "Price"],
            top=top,
        )
        return [f"{d['Name']} ({d['Category']}): ${d['Price']:.2f}" async for d in results]


def load_environment():
    env_path = load_env(required=True)
    print(f"Loaded .env from: {env_path}")


def get_project_endpoint():
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/")
    services_endpoint = os.getenv("AZURE_AI_SERVICES_ENDPOINT")
    if not services_endpoint:
        raise ValueError("Set AZURE_AI_PROJECT_ENDPOINT or AZURE_AI_SERVICES_ENDPOINT in .env")
    project_name = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{services_endpoint.rstrip('/')}/api/projects/{project_name}"


def with_price_text(document):
    # The numeric Price stays for filtering; the text copy lets the search tool show the
    # price to the model.
    return {**document, "Description": f"{document['Description']} Price: ${document['Price']:.2f}."}


async def get_search_connection(project_client):
    connection = await project_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH, include_credentials=True)
    api_key = getattr(connection.credentials, "api_key", None)
    if not api_key:
        raise RuntimeError(f"Connection {connection.name} has no API key; this lab expects key auth for AI Search.")
    print(f"Azure AI Search connection: {connection.name}")
    print(f"Azure AI Search endpoint:   {connection.target}")
    return connection, AzureKeyCredential(api_key)


async def create_fitness_index(index_client):
    fields = [
        SimpleField(name="FitnessItemID", type=SearchFieldDataType.String, key=True),
        SearchableField(name="Name", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="Category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="Price", type=SearchFieldDataType.Double, filterable=True, sortable=True, facetable=True),
        SearchableField(name="Description", type=SearchFieldDataType.String),
    ]
    existing = [name async for name in index_client.list_index_names()]
    if INDEX_NAME in existing:
        await index_client.delete_index(INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")
    created = await index_client.create_index(SearchIndex(name=INDEX_NAME, fields=fields))
    print(f"Created index: {created.name}")


async def upload_documents(search_client):
    results = await search_client.upload_documents(documents=[with_price_text(d) for d in SAMPLE_DOCUMENTS])
    failed = [r for r in results if not r.succeeded]
    if failed:
        raise RuntimeError("Upload failed: " + "; ".join(f"{r.key}: {r.error_message}" for r in failed))
    print(f"Uploaded {len(results)} documents.")


async def wait_for_documents(search_client, expected, timeout_seconds=30):
    # Uploaded documents become searchable after a short indexing delay.
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        count = await search_client.get_document_count()
        if count >= expected or asyncio.get_running_loop().time() > deadline:
            print(f"Searchable documents: {count}/{expected}")
            return count
        await asyncio.sleep(1)


async def print_direct_search(search_client, text):
    print(f"\nDirect Azure AI Search check for '{text}':")
    results = await search_client.search(search_text=text, top=10)
    async for doc in results:
        print(f"  {doc['Name']} | {doc['Category']} | ${doc['Price']:.2f} | {doc['Description']}")


async def print_tool_calls(project_client, thread_id):
    # Shows what each tool returned to the model in the latest run (e.g. which fields
    # the search tool passed on).
    runs = project_client.agents.runs.list(thread_id=thread_id, order=ListSortOrder.DESCENDING, limit=1)
    run = await anext(aiter(runs))
    offered = [t.function.name if t.type == "function" else t.type for t in run.tools or []]
    print(f"  Tools offered in run {run.id} ({run.status}): {offered}")
    print("  Tool calls:")
    async for step in project_client.agents.run_steps.list(
        thread_id=thread_id, run_id=run.id, order=ListSortOrder.ASCENDING
    ):
        for call in getattr(step.step_details, "tool_calls", None) or []:
            if call.type == "azure_ai_search":
                print(f"  - azure_ai_search input: {call.azure_ai_search.get('input')}")
                print(f"    azure_ai_search output: {call.azure_ai_search.get('output')}")
            elif call.type == "function":
                print(f"  - function {call.function.name}({call.function.arguments})")
                # The SDK model has no output attribute; the value is only in the raw
                # response data.
                print(f"    output: {call.function.get('output')}")
            else:
                print(f"  - {call.type}")


def load_settings():
    project_endpoint = get_project_endpoint()
    model = os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini"
    print(f"Foundry project endpoint: {project_endpoint}")
    print(f"Model: {model}")
    return project_endpoint, model


def create_search_clients(connection, search_key):
    index_client = SearchIndexClient(endpoint=connection.target, credential=search_key)
    search_client = SearchClient(endpoint=connection.target, index_name=INDEX_NAME, credential=search_key)
    return index_client, search_client


async def create_agent(project_client, model, connection):
    """Step 2: create the Foundry agent with the Azure AI Search tool on the index;
    return its definition."""
    search_tool = AzureAISearchTool(
        index_connection_id=connection.id,
        index_name=INDEX_NAME,
        query_type=AzureAISearchQueryType.SIMPLE,
        top_k=5,
    )
    definition = await project_client.agents.create_agent(
        model=model,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=search_tool.definitions,
        tool_resources=search_tool.resources,
    )
    print(f"Created agent: {definition.id}")
    return definition


async def ask(project_client, agent, thread, question):
    """Step 3: ask one question on the thread, print the answer and the tool calls;
    return the thread."""
    print(f"\n# User: {question}")
    response = await agent.get_response(messages=question, thread=thread)
    thread = response.thread
    print(f"# Agent: {response.message.content}")
    try:
        await print_tool_calls(project_client, thread.id)
    except Exception as exc:
        print(f"  Could not list tool calls: {exc!r}")
    return thread


async def cleanup(project_client, index_client, search_client, thread, definition, index_created):
    """Step 4: delete what the run created (thread, agent, index) and close the search
    clients."""
    if thread is not None and thread.id is not None:
        try:
            await thread.delete()
            print("Deleted thread")
        except Exception as exc:
            print(f"Could not delete thread: {exc}")
    if definition is not None:
        try:
            await project_client.agents.delete_agent(definition.id)
            print(f"Deleted agent: {definition.id}")
        except Exception as exc:
            print(f"Could not delete agent: {exc}")
    if index_created:
        try:
            await index_client.delete_index(INDEX_NAME)
            print(f"Deleted index: {INDEX_NAME}")
        except Exception as exc:
            print(f"Could not delete index: {exc}")
    await search_client.close()
    await index_client.close()


async def main():
    print("Disclaimer: this is a technical demonstration and does not provide medical advice.")
    load_environment()
    project_endpoint, model = load_settings()

    async with (
        DefaultAzureCredential() as credential,
        AzureAIAgent.create_client(credential=credential, endpoint=project_endpoint) as project_client,
    ):
        connection, search_key = await get_search_connection(project_client)
        index_client, search_client = create_search_clients(connection, search_key)
        index_created = False
        definition = None
        thread = None
        try:
            print("\n=== Step 1: create and fill the search index ===")
            await create_fitness_index(index_client)
            index_created = True
            await upload_documents(search_client)
            await wait_for_documents(search_client, len(SAMPLE_DOCUMENTS))
            await print_direct_search(search_client, "Strength")

            print("\n=== Step 2: create the agent (Azure AI Search tool + local price plugin) ===")
            definition = await create_agent(project_client, model, connection)
            # Semantic Kernel wraps the Foundry agent and adds the local price plugin.
            agent = AzureAIAgent(
                client=project_client, definition=definition, plugins=[FitnessCatalogPlugin(search_client)]
            )

            print("\n=== Step 3: ask the agent ===")
            thread = AzureAIAgentThread(client=project_client)
            for question in QUESTIONS:
                thread = await ask(project_client, agent, thread, question)
        finally:
            print("\n=== Cleanup ===")
            await cleanup(project_client, index_client, search_client, thread, definition, index_created)


if __name__ == "__main__":
    http_log.enable()
    asyncio.run(main())
