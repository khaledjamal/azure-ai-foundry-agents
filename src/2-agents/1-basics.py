"""Agent basics lab (1-basics.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/1-basics.ipynb

This script shows the building blocks of an AI agent in Azure AI Foundry's Agent Service: the agent (a model with
instructions, here a health advisor), a conversation thread, the user's messages, and a run, in which the agent reads
the thread and replies. Like Microsoft's notebook, it uses the classic agents API.

What it does:
1. Creates a health advisor agent.
2. Asks two questions on the same conversation thread, so the service keeps the conversation history instead of the
   script resending it. It checks that each run really completed and exits with an error if one didn't.
3. Prints the conversation.
4. Deletes the thread and the agent.

Connection method:
- Client: AIProjectClient from azure-ai-projects, created with the Foundry project endpoint and an Entra ID token from
  the Azure CLI login.
- Agents are only reachable through the project client (project_client.agents); a plain AzureOpenAI client cannot use
  them.

Run from the project root:
    uv run python src/2-agents/1-basics.py
"""

import os

from azure.ai.agents.models import ListSortOrder, RunStatus
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

AGENT_NAME = "fun-fit-health-advisor"

HEALTH_ADVISOR_INSTRUCTIONS = """
You are a friendly AI Health Advisor.
You provide general health, fitness, and nutrition information, but always:
1. Include medical disclaimers.
2. Encourage the user to consult healthcare professionals.
3. Provide general, non-diagnostic advice around wellness, diet, and fitness.
4. Clearly remind them you're not a doctor.
5. Encourage safe and balanced approaches to exercise and nutrition.
""".strip()

QUESTIONS = [
    "How do I calculate my BMI, and what does it mean?",
    "Can you give me a balanced meal plan for someone who exercises 3x a week?",
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


def print_conversation(project_client, thread_id):
    messages = project_client.agents.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
    for message in messages:
        for text_message in message.text_messages:
            print(f"\n{message.role.upper()}: {text_message.text.value}")


def main():
    print_section("AGENT BASICS - AIProjectClient")
    load_environment()
    model = require_env("MODEL_DEPLOYMENT_NAME")
    project_endpoint = get_project_endpoint()
    print("Connection method: AIProjectClient (project_client.agents)")
    print(f"Project endpoint:  {project_endpoint}")
    print(f"Model deployment:  {model}")

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project_client:
        agent = project_client.agents.create_agent(
            model=model,
            name=AGENT_NAME,
            instructions=HEALTH_ADVISOR_INSTRUCTIONS,
        )
        print(f"Created agent: {agent.name} ({agent.id})")

        thread = None
        failed = []
        try:
            thread = project_client.agents.threads.create()
            print(f"Created thread: {thread.id}")

            for number, question in enumerate(QUESTIONS, start=1):
                print_section(f"Question {number}")
                run = ask(project_client, thread.id, agent.id, question)
                # Azure reports a failed run as a status, not as a Python error, so check it here.
                if run.status != RunStatus.COMPLETED:
                    failed.append(number)

            print_section("Conversation (oldest to newest)")
            print_conversation(project_client, thread.id)
        finally:
            print_section("Cleanup")
            if thread:
                project_client.agents.threads.delete(thread.id)
                print(f"Deleted thread: {thread.id}")
            project_client.agents.delete_agent(agent.id)
            print(f"Deleted agent: {agent.id}")

    if failed:
        print_section(f"FINISHED WITH ERRORS: question(s) {failed} did not complete")
        raise SystemExit(1)
    print_section("DONE")


if __name__ == "__main__":
    http_log.enable()
    main()
