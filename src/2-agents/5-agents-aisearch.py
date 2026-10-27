"""Foundry agent with an Azure AI Search tool and a local price function tool
(5-agents-aisearch.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/5-agents-aisearch.ipynb

This script builds a shopping assistant for a small fitness product catalog. It uses the
classic agents API, as Microsoft's notebook does; 5-agents-aisearch-v2.py is the same
agent on the new agents API, compared on newer models.

What it does:
1. Builds a small fitness product index in Azure AI Search.
2. Gives a classic-API agent two tools:
   - Azure AI Search (through the project's Azure AI Search connection), to find
     products by description;
   - find_items_under(max_price, category), a price filter that runs in this script as a
     function tool, to answer budget questions exactly.
3. Asks the notebook's two questions and prints each run's status and every tool call,
   so you can see which tool the agent chose and what came back.
4. Deletes each question's thread after its answer, the agent and the index at the end.

Finding: the search tool passes only text fields to the model, so prices never reached
it and the agent said they weren't given; putting the price into the product text and
adding the price filter fixed it.

Settings from .env: AZURE_AI_SERVICES_ENDPOINT (or AZURE_AI_PROJECT_ENDPOINT),
AZURE_AI_PROJECT_NAME, MODEL_DEPLOYMENT_NAME, AZURE_AI_SEARCH_ENDPOINT,
AZURE_AI_SEARCH_API_KEY, optional AZURE_AI_SEARCH_CONNECTION_NAME. Paid: a few
gpt-4o-mini calls, well under $0.01 per run.

Run from the project root:
    uv run python src/2-agents/5-agents-aisearch.py
"""

import os
import sys
import time

from azure.ai.agents.models import (
    AzureAISearchQueryType,
    AzureAISearchTool,
    FunctionTool,
    ListSortOrder,
    RunStatus,
)
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)

from workshop_utils import http_log
from workshop_utils.settings import load_env

INDEX_NAME = "myfitnessindex"
AGENT_NAME = "fitness-agent-search"
DELETE_AGENT_AFTER_RUN = True
DELETE_INDEX_AFTER_RUN = True

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


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_environment():
    env_path = load_env(required=True)
    print(f"Loaded .env from: {env_path}")


def first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def require_any_env(*names):
    value = first_env(*names)
    if not value:
        joined = " or ".join(names)
        raise ValueError(f"Required environment variable is not set: {joined}")
    return value


def get_project_endpoint():
    direct_endpoint = first_env(
        "AZURE_AI_PROJECT_ENDPOINT",
        "PROJECT_ENDPOINT",
        "FOUNDRY_PROJECT_ENDPOINT",
    )
    if direct_endpoint:
        return direct_endpoint.rstrip("/")

    services_endpoint = require_any_env("AZURE_AI_SERVICES_ENDPOINT").rstrip("/")
    project_name = require_any_env("AZURE_AI_PROJECT_NAME")
    return f"{services_endpoint}/api/projects/{project_name}"


def get_model_name():
    return first_env("MODEL_DEPLOYMENT_NAME", "OPENAI_DEPLOYMENT_NAME") or "gpt-4o-mini"


def create_search_clients():
    search_endpoint = require_any_env("AZURE_AI_SEARCH_ENDPOINT", "SEARCH_ENDPOINT")
    search_api_key = require_any_env("AZURE_AI_SEARCH_API_KEY", "SEARCH_API_KEY")

    credential = AzureKeyCredential(search_api_key)

    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=credential,
    )

    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=INDEX_NAME,
        credential=credential,
    )

    print(f"Azure AI Search endpoint: {search_endpoint}")
    print("SearchIndexClient created successfully.")
    return index_client, search_client


def create_fitness_index(index_client):
    fields = [
        SimpleField(
            name="FitnessItemID",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="Name",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchableField(
            name="Category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="Price",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="Description",
            type=SearchFieldDataType.String,
        ),
    ]

    existing_names = {index.name for index in index_client.list_indexes()}
    if INDEX_NAME in existing_names:
        index_client.delete_index(INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")
        time.sleep(1)

    index = SearchIndex(name=INDEX_NAME, fields=fields)
    created = index_client.create_index(index)
    print(f"Created index: {created.name}")


def with_price_text(document):
    # The Azure AI Search tool only passes text fields to the model (it selects
    # FitnessItemID, Name, Category, Description), so the numeric Price never reached
    # it. The numeric Price stays for filtering and sorting.
    return {**document, "Description": f"{document['Description']} Price: ${document['Price']:.2f}."}


def upload_fitness_documents(search_client):
    results = search_client.upload_documents(documents=[with_price_text(d) for d in SAMPLE_DOCUMENTS])

    failures = []
    for result in results:
        status = "SUCCESS" if result.succeeded else "FAILED"
        print(f"Upload {status}: key={result.key}")
        if not result.succeeded:
            failures.append(result)

    if failures:
        messages = [f"{item.key}: {item.error_message}" for item in failures]
        raise RuntimeError("One or more documents failed to upload: " + "; ".join(messages))

    print(f"Uploaded {len(SAMPLE_DOCUMENTS)} fitness documents.")


def make_find_items_under(search_client):
    # Local function tool: the agent asks for it, and this script runs it (not Azure).
    # Same logic as the Semantic Kernel plugin in
    # src/4-frameworks/1-rag-sk-agents-aisearch.py.
    def products(filter_expression, top=10):
        results = search_client.search(
            search_text="*",
            filter=filter_expression,
            order_by=["Price asc"],
            select=["Name", "Category", "Price"],
            top=top,
        )
        return [f"{d['Name']} ({d['Category']}): ${d['Price']:.2f}" for d in results]

    def find_items_under(max_price: float, category: str | None = None) -> str:
        """Find fitness products priced at or below max_price, optionally in one category (Strength, Flexibility or Cardio); use it whenever the user gives a budget or price limit.

        :param max_price: Maximum price in US dollars.
        :param category: Product category, or empty for all categories.
        :return: Matching products with their prices.
        """  # noqa: W505 - the agent gets this docstring as the tool description
        category = category.strip().title() if category and category.strip() else None
        category_filter = f" and Category eq '{category.replace(chr(39), chr(39) * 2)}'" if category else ""

        matches = products(f"Price le {max_price}{category_filter}")
        print(f"  [function] find_items_under(max_price={max_price}, category={category}) -> {len(matches)} item(s)")
        if matches:
            return "\n".join(matches)

        answer = f"No products cost ${max_price:.2f} or less" + (f" in category {category}." if category else ".")
        if category:
            cheapest = products(category_filter.removeprefix(" and "), top=1)
            if cheapest:
                answer += f" Cheapest product in {category}: {cheapest[0]}"
        return answer

    return find_items_under


def verify_search_index(search_client):
    print("\nDirect Azure AI Search check for 'Strength':")
    print("-" * 80)

    last_error = None
    for attempt in range(1, 6):
        try:
            results = list(search_client.search(search_text="Strength", top=10))
            if results:
                for doc in results:
                    print(
                        f"{doc.get('FitnessItemID')}: {doc.get('Name')} | "
                        f"{doc.get('Category')} | ${doc.get('Price')} | "
                        f"{doc.get('Description')}"
                    )
                print("-" * 80)
                return
        except Exception as exc:
            last_error = exc

        if attempt < 5:
            time.sleep(2)

    if last_error:
        raise RuntimeError(f"Could not verify uploaded documents: {last_error}")
    raise RuntimeError("The search index query returned no documents after multiple attempts.")


def create_project_client():
    project_endpoint = get_project_endpoint()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )

    print(f"Foundry project endpoint: {project_endpoint}")
    print("AIProjectClient created successfully.")
    return project_client


def resolve_ai_search_connection(project_client):
    explicit_name = first_env(
        "AZURE_AI_SEARCH_CONNECTION_NAME",
        "AI_SEARCH_CONNECTION_NAME",
    )

    if explicit_name:
        try:
            connection = project_client.connections.get(connection_name=explicit_name)
        except TypeError:
            connection = project_client.connections.get(explicit_name)

        print(f"Azure AI Search connection name: {connection.name}")
        print(f"Azure AI Search connection ID:   {connection.id}")
        return connection

    try:
        connection = project_client.connections.get_default(connection_type=ConnectionType.AZURE_AI_SEARCH)
        print(f"Azure AI Search connection name: {connection.name}")
        print(f"Azure AI Search connection ID:   {connection.id}")
        return connection
    except Exception as default_error:
        print(f"No default Azure AI Search connection was found: {default_error}")

    connections = list(project_client.connections.list())
    print(f"Project currently has {len(connections)} connection(s).")
    for connection in connections:
        print(
            f"  Name: {getattr(connection, 'name', None)} | "
            f"Type: {getattr(connection, 'connection_type', None)} | "
            f"ID: {getattr(connection, 'id', None)}"
        )

    for connection in connections:
        connection_type = getattr(connection, "connection_type", None)
        if connection_type == ConnectionType.AZURE_AI_SEARCH:
            print(f"Using Azure AI Search connection: {connection.name}")
            return connection

    raise RuntimeError(
        "No current Foundry project connection to Azure AI Search was found.\n"
        "Add your Azure AI Search resource to this Foundry project under Connected resources, "
        "then rerun this script.\n"
        "Optionally set AZURE_AI_SEARCH_CONNECTION_NAME in .env to the connection name."
    )


def create_search_agent(project_client, search_connection, search_client):
    model_name = get_model_name()

    ai_search_tool = AzureAISearchTool(
        index_connection_id=search_connection.id,
        index_name=INDEX_NAME,
        query_type=AzureAISearchQueryType.SIMPLE,
        top_k=5,
        filter="",
    )

    # runs.create_and_process calls the function locally whenever the model asks for it.
    price_tool = FunctionTool({make_find_items_under(search_client)})
    project_client.agents.enable_auto_function_calls(price_tool)

    agent = project_client.agents.create_agent(
        model=model_name,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=ai_search_tool.definitions + price_tool.definitions,
        tool_resources=ai_search_tool.resources,
    )

    print(f"Created Azure AI Search agent: {agent.id}")
    print(f"Model: {model_name}")
    print(f"Tools: {[getattr(tool, 'type', None) for tool in agent.tools]} (function: find_items_under)")
    return agent


def print_thread_messages(project_client, thread_id):
    messages = project_client.agents.messages.list(
        thread_id=thread_id,
        order=ListSortOrder.ASCENDING,
    )

    data = getattr(messages, "data", messages)

    print("\nConversation:")
    print("-" * 80)

    for message in data:
        role = getattr(message.role, "value", str(message.role)).upper()
        text_messages = getattr(message, "text_messages", None) or []

        if text_messages:
            for text_message in text_messages:
                print(f"\n{role}:\n{text_message.text.value}")
        else:
            content = getattr(message, "content", None)
            if content:
                print(f"\n{role}:\n{content}")

    print("-" * 80)


def print_tool_calls(project_client, thread_id, run):
    # Shows which tools the model actually called in this run, so a skipped search is
    # visible.
    print("Tool calls:")
    calls = 0
    for step in project_client.agents.run_steps.list(thread_id=thread_id, run_id=run.id, order=ListSortOrder.ASCENDING):
        for call in getattr(step.step_details, "tool_calls", None) or []:
            calls += 1
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
    if not calls:
        print("  (none)")


def run_agent_query(project_client, agent, question):
    thread = project_client.agents.threads.create()
    print(f"Created thread: {thread.id}")
    print(f"Question: {question}")

    try:
        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=question,
        )
        print(f"Created user message: {message.id}")

        run = project_client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        print(f"Run status: {run.status}")

        if getattr(run, "last_error", None):
            print(f"Run error: {run.last_error}")

        try:
            print_tool_calls(project_client, thread.id, run)
        except Exception as exc:
            print(f"Could not list tool calls: {exc!r}")

        print_thread_messages(project_client, thread.id)
        return run
    finally:
        # Each question gets its own thread, so delete it here rather than in the final
        # cleanup.
        try:
            project_client.agents.threads.delete(thread.id)
            print(f"Deleted thread: {thread.id}")
        except Exception as exc:
            print(f"Could not delete thread {thread.id}: {exc}")


def cleanup_agent(project_client, agent):
    if agent is None:
        return
    project_client.agents.delete_agent(agent.id)
    print(f"Deleted temporary agent: {agent.id}")


def cleanup_index(index_client):
    if index_client is None:
        return

    existing_names = {index.name for index in index_client.list_indexes()}
    if INDEX_NAME in existing_names:
        index_client.delete_index(INDEX_NAME)
        print(f"Deleted temporary index: {INDEX_NAME}")


def run_questions(project_client, agent):
    """Step 3: ask each question on its own thread; return (number, status, error) for
    each run that didn't complete."""
    failed_runs = []
    for index, question in enumerate(QUESTIONS, start=1):
        print_section(f"QUESTION {index} OF {len(QUESTIONS)}")
        run = run_agent_query(project_client, agent, question)
        # Azure reports a failed run as a status, not as a Python error, so check it
        # here.
        if run.status != RunStatus.COMPLETED:
            failed_runs.append((index, run.status, getattr(run, "last_error", None)))
    return failed_runs


def check_results(failed_runs):
    """Every agent run must have completed; print which ones didn't."""
    print("\n" + "=" * 80)
    if failed_runs:
        print(f"FINISHED WITH ERRORS: {len(failed_runs)} of {len(QUESTIONS)} agent runs did not complete")
        print("=" * 80)
        for index, status, error in failed_runs:
            print(f"Question {index}: status {status}, error: {error}")
        return False
    print(f"COMPLETED: all {len(QUESTIONS)} agent runs finished with status COMPLETED")
    print("=" * 80)
    return True


def cleanup(project_client, agent, index_client):
    """Step 4: delete the agent and the index (each question's thread is already
    deleted), close the project client."""
    if DELETE_AGENT_AFTER_RUN and project_client is not None and agent is not None:
        try:
            cleanup_agent(project_client, agent)
        except Exception as exc:
            print(f"Could not delete temporary agent: {exc}")

    if DELETE_INDEX_AFTER_RUN and index_client is not None:
        try:
            cleanup_index(index_client)
        except Exception as exc:
            print(f"Could not delete temporary index: {exc}")

    if project_client is not None:
        try:
            project_client.close()
        except Exception as exc:
            print(f"Could not close the project client: {exc}")


def main():
    print("=" * 80)
    print("AZURE AI SEARCH + FOUNDRY AGENT FITNESS DEMO")
    print("=" * 80)
    print("Disclaimer: This program is a technical demonstration and does not provide professional medical advice.")

    load_environment()

    index_client = None
    search_client = None
    project_client = None
    agent = None
    passed = False

    try:
        print_section("STEP 1: CREATE AND POPULATE AZURE AI SEARCH INDEX")
        index_client, search_client = create_search_clients()
        create_fitness_index(index_client)
        upload_fitness_documents(search_client)
        verify_search_index(search_client)

        print_section("STEP 2: CREATE FOUNDRY AGENT WITH AZURE AI SEARCH TOOL AND PRICE FUNCTION")
        project_client = create_project_client()
        search_connection = resolve_ai_search_connection(project_client)
        agent = create_search_agent(project_client, search_connection, search_client)

        print_section("STEP 3: RUN AGENT QUERIES")
        failed_runs = run_questions(project_client, agent)
        passed = check_results(failed_runs)

    finally:
        print_section("CLEANUP")
        cleanup(project_client, agent, index_client)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    http_log.enable()
    main()
