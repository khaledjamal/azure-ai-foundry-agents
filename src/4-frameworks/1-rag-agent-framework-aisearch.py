# src/4-frameworks/1-rag-agent-framework-aisearch.py
"""Microsoft Agent Framework + Azure AI Search: the Semantic Kernel lab's shopping
assistant written again the way Microsoft now recommends.

Workshop notebook it started from (through 1-rag-sk-agents-aisearch.py):
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/1-rag-sk-agents-aisearch.ipynb

Microsoft calls Agent Framework the direct successor of Semantic Kernel (and of
AutoGen). This script keeps everything of 1-rag-sk-agents-aisearch.py that isn't
Semantic Kernel itself, so the two read side by side.

What it does:
1. Builds a small fitness product index in Azure AI Search.
2. Builds one Agent Framework Agent with two tools: the Azure AI Search tool on the
   index, and find_items_under, a plain Python function that runs in this script.
3. Asks two shopping questions in one session.
4. Deletes the index.

Same as 1-rag-sk-agents-aisearch.py:
- The catalog, the index fields, the two questions and the instructions.
- The price in each product's text ("Price: $499.00."): the search tool passes only
  text fields to the model.
- The price filter find_items_under, which queries the index with a real filter
  (Price le <max>), so a budget question gets an exact answer.
- After each model call the script prints the tool calls and what the search tool
  returned.

What Agent Framework changes:
- One Agent on a FoundryChatClient: no agent is created in Foundry Agent Service
  first and then wrapped, and there is no Kernel. So cleanup has no agent or thread
  to delete.
- The model is called through the Responses API of the Foundry project
  (azure-ai-projects 2.x), not through threads and runs, so the old
  `azure-ai-projects<2` pin of the Semantic Kernel environment is gone.
- find_items_under is a plain function: its name becomes the tool name, its
  docstring the description and its Annotated hints the parameter descriptions; no
  plugin class or decorator. Agent Framework runs it when the model asks and sends
  the result back.
- agent.create_session() and agent.run(question, session=session) replace
  AzureAIAgentThread and get_response().
- A chat middleware (print_tool_calls) sees every model response, which is where
  the tool calls are printed.
- The Azure AI Search tool comes from FoundryChatClient.get_azure_ai_search_tool(),
  marked experimental in agent-framework-foundry 1.16.2.
- The Responses API stores each response in the Foundry project by default (the
  session continues from the previous one); this script doesn't delete them.

Connection method:
- Client: AIProjectClient (async) from azure-ai-projects 2.x with the Foundry project
  endpoint and an Entra ID token from the Azure CLI login; FoundryChatClient uses the
  same client.
- FoundryChatClient opens its own OpenAI client for the Responses API; `async with`
  on the Agent closes it.
- Azure AI Search: the endpoint and key come from the project's default Azure AI Search
  connection; the search tool reaches the index through the same connection.

Uses its own environment (requirements-agent-framework.txt). Run from the project
root:
    .venv-agent-framework/bin/python src/4-frameworks/1-rag-agent-framework-aisearch.py
"""

import asyncio
import os
from typing import Annotated

from agent_framework import Agent, chat_middleware
from agent_framework.foundry import FoundryChatClient
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)

from workshop_utils import http_log
from workshop_utils.settings import load_env

INDEX_NAME = "fitness-af-index"
AGENT_NAME = "fitness-af-agent"

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
        "Name": "Exercise Mat",
        "Category": "Flexibility",
        "Price": 25.0,
        "Description": "Non-slip mat for stretching, core workouts and floor exercises.",
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

# The fitness index's fields; the documents in SAMPLE_DOCUMENTS use these names.
FITNESS_INDEX_FIELDS = [
    SimpleField(name="FitnessItemID", type=SearchFieldDataType.String, key=True),
    SearchableField(name="Name", type=SearchFieldDataType.String, filterable=True),
    # The lowercase normalizer makes the Category filter ignore case ("cardio"
    # finds "Cardio"), whatever spelling the model sends. SearchField, not
    # SearchableField: the shortcut drops normalizer_name.
    SearchField(
        name="Category",
        type=SearchFieldDataType.String,
        searchable=True,
        filterable=True,
        facetable=True,
        sortable=False,
        normalizer_name="lowercase",
    ),
    SimpleField(name="Price", type=SearchFieldDataType.Double, filterable=True, sortable=True, facetable=True),
    SearchableField(name="Description", type=SearchFieldDataType.String),
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

# These calls happen inside one line of this lab (get_default() or agent.run()), so no
# print can go right before each one; http_log prints each description before a
# matching call that it logs in full. The first match wins, and [?] is a literal
# question mark.
CALL_DESCRIPTIONS = {
    "GET */connections[?]*": "get_default(): list the connections of the Foundry "
    "project, to find the default Azure AI Search connection.",
    "POST */getConnectionWithCredentials*": "get_default(): read the key of the "
    "default Azure AI Search connection from the Foundry project.",
    "POST */openai/v1/responses*": "agent.run(): Agent Framework calls the model "
    "through the Responses API of the Foundry project, with the instructions, the "
    "tools and the question, or with find_items_under's result after the model "
    "asked for it.",
}


def escape_quotes_for_search_filter(value):
    """Double each single quote in value, for a quoted Azure AI Search filter value.

    - Azure AI Search filters use the OData query language (a web standard); in it,
      a single quote inside a quoted value is written twice: O'Brien -> O''Brien.
    - Only the single quote needs this: the SDK sends the filter in a JSON body,
      which takes care of other characters.
    - Why: the category comes from the model, so a stray quote must not end the
      value early and change the filter.
    """
    return value.replace("'", "''")


def make_find_items_under(search_client):
    """Return the find_items_under tool function, bound to the fitness index's search
    client (a local tool: it runs in this script, not in Azure)."""

    async def search_products(filter_expression, top=10):
        """Search the fitness index with an OData filter; "Name (Category): $price"
        lines, cheapest first."""
        print(
            "\n>>> find_items_under function: search the fitness index in Azure AI "
            f"Search with this filter: {filter_expression}"
        )
        results = await search_client.search(
            search_text="*",
            filter=filter_expression,
            order_by=["Price asc"],
            select=["Name", "Category", "Price"],
            top=top,
        )

        return [f"{document['Name']} ({document['Category']}): ${document['Price']:.2f}" async for document in results]

    async def find_items_under(
        max_price: Annotated[float, "Maximum price in US dollars"],
        category: Annotated[str | None, "Product category, or empty for all categories"] = None,
    ) -> str:
        """Find fitness products whose price is at or below a maximum price,
        optionally in one category (Strength, Flexibility or Cardio). Use it
        whenever the user gives a budget or price limit."""
        category_filter = f" and Category eq '{escape_quotes_for_search_filter(category)}'" if category else ""

        matches = await search_products(f"Price le {max_price}{category_filter}")
        print(f"  [tool] find_items_under(max_price={max_price}, category={category}) -> {len(matches)} item(s)")

        if matches:
            return "\n".join(matches)

        answer = f"No products cost ${max_price:.2f} or less" + (f" in category {category}." if category else ".")

        if category:
            cheapest = await search_products(category_filter.removeprefix(" and "), top=1)
            if cheapest:
                answer += f" Cheapest product in {category}: {cheapest[0]}"

        return answer

    return find_items_under


@chat_middleware
async def print_tool_calls(context, call_next):
    """Print the tool calls in each model response, and what the search tool returned.

    - Agent Framework keeps function calls in its own messages but leaves the Azure AI
      Search tool's items out, so they are read from the raw Responses API response.
    - azure_ai_search_call is the search; its result comes back as a separate
      azure_ai_search_call_output item.
    """
    await call_next()

    response = context.result.raw_representation
    non_tool_types = ("message", "reasoning")
    tool_items = [item for item in response.output if item.type not in non_tool_types]
    print(f"  Model response {response.id}: {len(tool_items)} tool item(s)")

    for item in tool_items:
        if item.type == "function_call":
            print(f"  - function {item.name}({item.arguments})")
        elif item.type == "azure_ai_search_call":
            print(f"  - azure_ai_search input: {getattr(item, 'arguments', None)}")
        elif item.type == "azure_ai_search_call_output":
            print(f"    azure_ai_search output: {getattr(item, 'output', None)}")
        else:
            print(f"  - {item.type}")


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


async def delete_fitness_index_if_exists(index_client):
    print("\n>>> List the indexes in Azure AI Search, to see whether the fitness index already exists.")
    existing = [name async for name in index_client.list_index_names()]

    if INDEX_NAME in existing:
        print("\n>>> Delete the old fitness index from Azure AI Search.")
        await index_client.delete_index(INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")


async def recreate_fitness_index(index_client):
    """Delete the fitness index if it exists, then create it again."""
    await delete_fitness_index_if_exists(index_client)

    print("\n>>> Create the fitness index in Azure AI Search.")
    created = await index_client.create_index(SearchIndex(name=INDEX_NAME, fields=FITNESS_INDEX_FIELDS))
    print(f"Created index: {created.name}")


async def upload_documents(search_client):
    documents_with_price_text = [with_price_text(d) for d in SAMPLE_DOCUMENTS]

    print("\n>>> Upload the fitness product documents to the fitness index in Azure AI Search.")
    results = await search_client.upload_documents(documents=documents_with_price_text)
    failed = [r for r in results if not r.succeeded]
    if failed:
        failures = [f"{result.key}: {result.error_message}" for result in failed]
        raise RuntimeError("Upload failed: " + "; ".join(failures))

    print(f"Uploaded {len(results)} documents.")


async def wait_for_documents(search_client, expected, timeout_seconds=30):
    # Uploaded documents become searchable after a short indexing delay.
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    print(
        "\n>>> Ask Azure AI Search how many documents the fitness index can search, "
        "until all of them can. The first check is logged in full, later ones as one "
        "`HTTP #n GET` line."
    )

    while True:
        count = await search_client.get_document_count()

        if count >= expected or asyncio.get_running_loop().time() > deadline:
            print(f"Searchable documents: {count}/{expected}")
            return count

        await asyncio.sleep(1)  # seconds


async def print_direct_search(search_client, text):
    print(f"\nDirect Azure AI Search check for '{text}':")

    print(f"\n>>> Search the fitness index in Azure AI Search for '{text}', without the agent.")
    results = await search_client.search(search_text=text, top=10)

    async for doc in results:
        print(f"  {doc['Name']} | {doc['Category']} | ${doc['Price']:.2f} | {doc['Description']}")


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


def create_agent(project_client, model, connection, search_client):
    """Step 2: build the Agent with the Azure AI Search tool on the index and the local
    find_items_under function. Nothing is created in Azure: the tools and instructions
    go with each model call. The caller opens and closes it with `async with`."""
    chat_client = FoundryChatClient(project_client=project_client, model=model)
    search_tool = chat_client.get_azure_ai_search_tool(
        index_connection_id=connection.id,
        index_name=INDEX_NAME,
        query_type="simple",
        top_k=5,  # results the tool passes to the model per search
    )

    return Agent(
        client=chat_client,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=[search_tool, make_find_items_under(search_client)],
        middleware=[print_tool_calls],
    )


async def ask(agent, session, question):
    """Step 3: ask one question in the session and print the answer."""
    print(f"\n# User: {question}")

    print(
        "\n>>> Run agent.run(), in which Agent Framework makes one or more "
        "Responses API calls: each is described just before it in this log."
    )
    response = await agent.run(question, session=session)
    print(f"# Agent: {response.text}")


async def cleanup(index_client, search_client, index_created):
    """Step 4: delete the index and close the search clients."""
    if index_created:
        try:
            print("\n>>> Delete the fitness index from Azure AI Search.")
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
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
    ):
        connection, search_key = await get_search_connection(project_client)
        index_client, search_client = create_search_clients(connection, search_key)

        index_created = False

        try:
            print("\n=== Step 1: create and fill the search index ===")
            await recreate_fitness_index(index_client)
            index_created = True
            await upload_documents(search_client)
            await wait_for_documents(search_client, len(SAMPLE_DOCUMENTS))

            await print_direct_search(search_client, "Strength")

            print("\n=== Step 2: build the agent (Azure AI Search tool + local price function) ===")
            async with create_agent(project_client, model, connection, search_client) as agent:
                print("\n=== Step 3: ask the agent ===")
                session = agent.create_session()

                for question in QUESTIONS:
                    await ask(agent, session, question)
        finally:
            print("\n=== Cleanup ===")
            await cleanup(index_client, search_client, index_created)


if __name__ == "__main__":
    http_log.enable(call_descriptions=CALL_DESCRIPTIONS)
    asyncio.run(main())
