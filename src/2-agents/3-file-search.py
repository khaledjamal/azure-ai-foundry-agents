"""File search lab (3-file-search.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/3-file-search.ipynb

This script gives an agent the File Search tool, so it answers from your own documents. It is the managed form of RAG,
with no search service to set up.

What it does:
1. Uploads two short Markdown files (recipes and dietary guidelines) into a vector store, a search index the Agent
   Service manages.
2. Creates a health resource agent with the File Search tool.
3. Asks three questions on one thread, such as a gluten-free lunch recipe, and checks that each run completed.
4. Prints the answers, each with citations naming the file it used, so a reader can check where it came from.
5. Deletes everything it created in the project: the thread, the agent, the vector store and the uploaded files.

Connection method: AIProjectClient from azure-ai-projects, created with the Foundry project
endpoint and an Entra ID token from the Azure CLI login.

Run from the project root:
    uv run python src/2-agents/3-file-search.py
"""

import os
import tempfile
from pathlib import Path

from azure.ai.agents.models import FilePurpose, FileSearchTool, ListSortOrder, RunStatus
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

AGENT_NAME = "health-search-agent"
VECTOR_STORE_NAME = "health_resources_example"

AGENT_INSTRUCTIONS = """
You are a health resource advisor with access to dietary and recipe files.
You:
1. Always present disclaimers (you're not a doctor!)
2. Provide references to the files when possible
3. Focus on general nutrition or recipe tips.
4. Encourage professional consultation for more detailed advice.
""".strip()

SAMPLE_FILES = {
    "recipes.md": """# Healthy Recipes Database

## Gluten-Free Recipes
1. Quinoa Bowl
   - Ingredients: quinoa, vegetables, olive oil
   - Instructions: Cook quinoa, add vegetables

2. Rice Pasta with Vegetables
   - Ingredients: rice pasta, mixed vegetables
   - Instructions: Boil pasta, sauté vegetables

## Diabetic-Friendly Recipes
1. Low-Carb Stir Fry
   - Ingredients: chicken, vegetables, tamari sauce
   - Instructions: Cook chicken, add vegetables

2. Greek Salad
   - Ingredients: cucumber, tomatoes, feta, olives
   - Instructions: Chop vegetables, combine

## Heart-Healthy Recipes
1. Baked Salmon
   - Ingredients: salmon, lemon, herbs
   - Instructions: Season salmon, bake

2. Mediterranean Bowl
   - Ingredients: chickpeas, vegetables, tahini
   - Instructions: Combine ingredients
""",
    "guidelines.md": """# Dietary Guidelines

## General Guidelines
- Eat a variety of foods
- Control portion sizes
- Stay hydrated

## Special Diets
1. Gluten-Free Diet
   - Avoid wheat, barley, rye
   - Focus on naturally gluten-free foods

2. Diabetic Diet
   - Monitor carbohydrate intake
   - Choose low glycemic foods

3. Heart-Healthy Diet
   - Limit saturated fats
   - Choose lean proteins
""",
}

QUESTIONS = [
    "Could you suggest a gluten-free lunch recipe?",
    "Show me some heart-healthy meal ideas.",
    "What guidelines do you have for someone with diabetes?",
]


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


def load_settings():
    model = require_env("MODEL_DEPLOYMENT_NAME")
    project_endpoint = get_project_endpoint()
    print("Connection method: AIProjectClient (project_client.agents)")
    print(f"Project endpoint:  {project_endpoint}")
    print(f"Model deployment:  {model}")
    return model, project_endpoint


def upload_sample_files(project_client):
    uploaded = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, content in SAMPLE_FILES.items():
            path = Path(temp_dir) / name
            path.write_text(content, encoding="utf-8")
            info = project_client.agents.files.upload_and_poll(file_path=str(path), purpose=FilePurpose.AGENTS)
            uploaded[info.id] = name
            print(f"Uploaded: {name} -> {info.id}")
    return uploaded


def build_vector_store(project_client, file_ids):
    """Step 1: put the uploaded files into a vector store, a search index the Agent Service manages; return it."""
    vector_store = project_client.agents.vector_stores.create_and_poll(file_ids=file_ids, name=VECTOR_STORE_NAME)
    counts = vector_store.file_counts
    print(
        f"Created vector store '{VECTOR_STORE_NAME}': {vector_store.id} "
        f"(files completed {counts.completed}, failed {counts.failed})"
    )
    return vector_store


def create_agent(project_client, model, vector_store):
    """Step 2: create the agent with the File Search tool on the vector store."""
    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    agent = project_client.agents.create_agent(
        model=model,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=file_search_tool.definitions,
        tool_resources=file_search_tool.resources,
    )
    print(f"Created agent: {agent.name} ({agent.id})")
    return agent


def create_thread(project_client):
    thread = project_client.agents.threads.create()
    print(f"Created thread: {thread.id}")
    return thread


def ask_questions(project_client, thread, agent):
    """Step 3: ask each question on the thread; return the numbers of the questions whose run didn't complete."""
    failed = []
    for number, question in enumerate(QUESTIONS, start=1):
        print_section(f"{number + 2}. Question {number}")
        run = ask(project_client, thread.id, agent.id, question)
        # Azure reports a failed run as a status, not as a Python error, so check it here.
        if run.status != RunStatus.COMPLETED:
            failed.append(number)
    return failed


def ask(project_client, thread_id, agent_id, question):
    print(f"User: {question}")
    project_client.agents.messages.create(thread_id=thread_id, role="user", content=question)
    run = project_client.agents.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)
    print(f"Run status: {run.status}")
    if run.usage:
        print(f"(tokens: prompt {run.usage.prompt_tokens}, completion {run.usage.completion_tokens})")
    if run.last_error:
        print(f"Run error: {run.last_error}")
    return run


def print_conversation(project_client, thread_id, file_names):
    messages = project_client.agents.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
    for message in messages:
        for text_message in message.text_messages:
            print(f"\n{message.role.upper()}: {text_message.text.value}")
        citations = {c.text: c.file_citation.file_id for c in message.file_citation_annotations}
        for marker, file_id in citations.items():
            print(f"  citation {marker} -> {file_names.get(file_id, file_id)}")


def cleanup(project_client, thread, agent, vector_store, uploaded):
    """Step 5: delete what the run created (whatever got created before an error): thread, agent, store, files."""
    if thread:
        project_client.agents.threads.delete(thread.id)
        print(f"Deleted thread: {thread.id}")
    if agent:
        project_client.agents.delete_agent(agent.id)
        print(f"Deleted agent: {agent.id}")
    if vector_store:
        project_client.agents.vector_stores.delete(vector_store.id)
        print(f"Deleted vector store: {vector_store.id}")
    for file_id, name in uploaded.items():
        project_client.agents.files.delete(file_id)
        print(f"Deleted file from the project: {name} ({file_id})")


def check_results(failed):
    """Every question's run must have completed."""
    if failed:
        print_section(f"FINISHED WITH ERRORS: question(s) {failed} did not complete")
        return False
    print_section("DONE")
    return True


def main():
    print_section("FILE SEARCH AGENT - AIProjectClient")
    load_environment()
    model, project_endpoint = load_settings()

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project_client:
        uploaded = {}
        failed = []
        vector_store = agent = thread = None
        try:
            print_section("1. Upload files and build the vector store")
            uploaded = upload_sample_files(project_client)
            vector_store = build_vector_store(project_client, list(uploaded))

            print_section("2. Create the agent")
            agent = create_agent(project_client, model, vector_store)
            thread = create_thread(project_client)
            failed = ask_questions(project_client, thread, agent)

            print_section("Conversation with citations (oldest to newest)")
            print_conversation(project_client, thread.id, uploaded)
        finally:
            print_section("Cleanup")
            cleanup(project_client, thread, agent, vector_store, uploaded)

    passed = check_results(failed)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
