"""Foundry agent (new agents API, azure-ai-projects 2.x) that uses the Azure Functions
MCP server's workout tools.

Replaces Microsoft's workshop notebook, which gave the agent an Azure Functions tool
through Storage queues (that needs the standard agent setup); here the Function app is a
remote MCP server instead:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/6-agents-az-functions.ipynb

This script connects a Foundry agent to tools hosted outside Azure AI Foundry through
MCP (Model Context Protocol), the open standard for giving agents tools. The tools run
on a small Azure Functions app (src/functions-mcp-server): one lists workout goals, the
other plans a workout.

What it does:
1. Checks the Function app with one request without a key, before creating the agent:
   HTTP 403 means it is stopped (the script prints the start command and exits, instead
   of failing midway); HTTP 401 means it is running and wants the key, as expected. That
   request also wakes the app, so the agent's first MCP call doesn't wait for a cold
   start.
2. Creates the agent. Its MCP tool points to the Function app's endpoint and
   authenticates through the project connection `workout-mcp-key` (Custom keys, key name
   x-functions-key), so the server's key sits in a Foundry project connection, not in
   the agent definition or the code.
3. Runs two turns in one conversation:
   - "Which workout goals can you plan for?"
     -> expects a list_workout_goals call
   - "Plan a 30-minute strength workout for me."
     -> expects suggest_workout(goal=strength, minutes=30)
   Every tool call needs approval: the script approves only list_workout_goals and
   suggest_workout on this server and denies anything else, so the agent can't call
   anything else on its own.
4. Deletes the agent version at the end.
Exits with 1 unless both calls happened and both turns got an answer.

Settings from .env: AZURE_FUNCTIONS_MCP_URL, AZURE_AI_SERVICES_ENDPOINT (or
AZURE_AI_PROJECT_ENDPOINT), AZURE_AI_PROJECT_NAME, MODEL_DEPLOYMENT_NAME,
AZURE_RESOURCE_GROUP (only for the start hint). A few small paid calls on gpt-4o-mini,
well under $0.01 per run.

Needs azure-ai-projects>=2 (in the separate agents venv). Run from the project root:
    source ../../.venv-agents-v2/bin/activate
    python src/2-agents/6-agents-az-functions-mcp.py [--model gpt-4o-mini]
"""

import argparse
import json
import os
from urllib.parse import urlparse

import requests
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MCPTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

AGENT_NAME = "workout-mcp-agent"
SERVER_LABEL = "workout_tools"
CONNECTION_NAME = "workout-mcp-key"
ALLOWED_TOOLS = ["list_workout_goals", "suggest_workout"]
INSTRUCTIONS = (
    "You help people plan simple home workouts. Use the workout tools for every answer: list_workout_goals to see "
    "which goals exist, suggest_workout to build a plan. Never invent a plan or a goal the tools didn't return. "
    "Show times, counts and exercises only as the tool gives them: don't split a block's minutes across its items "
    "or add numbers of your own. Present the plan clearly and end with one line saying it is general fitness "
    "information, not medical advice."
)
QUESTIONS = ["Which workout goals can you plan for?", "Plan a 30-minute strength workout for me."]
MAX_APPROVAL_ROUNDS = 4


def get_project_endpoint():
    direct_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if direct_endpoint:
        return direct_endpoint.rstrip("/")
    services_endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/")
    project_name = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{services_endpoint}/api/projects/{project_name}"


def check_function_app(url):
    """Return True if the Function app is running (it answers 401 without a key); print
    why not otherwise."""
    app_name = urlparse(url).hostname.split(".")[0]
    resource_group = os.getenv("AZURE_RESOURCE_GROUP") or "<resource-group>"
    try:
        response = requests.post(url, json={}, timeout=60)
    except requests.RequestException as error:
        print(f"Function app not reachable: {error}")
        return False
    if response.status_code == 403 and "stopped" in response.text:
        print("Function app is stopped. Start it with:")
        print(f"    az functionapp start -g {resource_group} -n {app_name}")
        return False
    if response.status_code == 401:
        print("Function app is running (HTTP 401 without a key, as expected)")
        return True
    print(f"Unexpected answer from the Function app: HTTP {response.status_code}")
    return False


def show_output(response, totals):
    """Print the MCP items of one response; return the approval requests and the
    successful MCP calls."""
    approvals, calls = [], []
    for item in response.output:
        if item.type == "mcp_list_tools":
            names = [tool.name for tool in item.tools]
            print(f"  mcp_list_tools  server={item.server_label}  tools={names}  error={item.error}")
        elif item.type == "mcp_approval_request":
            print(f"  mcp_approval_request  server={item.server_label}  tool={item.name}  arguments={item.arguments}")
            approvals.append(item)
        elif item.type == "mcp_call":
            print(f"  mcp_call  server={item.server_label}  tool={item.name}  arguments={item.arguments}")
            print(f"            output: {item.output}  error={item.error}")
            if item.output and not item.error:
                calls.append(item)
        elif item.type == "message":
            print("  message (answer, printed below)")
        else:
            print(f"  {item.type}")
    if response.usage:
        totals["input"] += response.usage.input_tokens
        totals["output"] += response.usage.output_tokens
        print(f"  (tokens: input {response.usage.input_tokens}, output {response.usage.output_tokens})")
    return approvals, calls


def ask(openai_client, agent_reference, question, previous_response_id, totals):
    """Send one user turn, answer its approval requests, and return the last response
    and the successful calls."""
    print(f"\nUser: {question}")
    follow_up = {"previous_response_id": previous_response_id} if previous_response_id else {}
    response = openai_client.responses.create(input=question, extra_body=agent_reference, **follow_up)
    print(f"Response (status {response.status}):")
    approvals, calls = show_output(response, totals)
    for _ in range(MAX_APPROVAL_ROUNDS):
        if not approvals:
            break
        replies = []
        for request in approvals:
            approve = request.server_label == SERVER_LABEL and request.name in ALLOWED_TOOLS
            print(f"  -> {'approve' if approve else 'deny'} {request.name}")
            replies.append({"type": "mcp_approval_response", "approve": approve, "approval_request_id": request.id})
        response = openai_client.responses.create(
            input=replies, previous_response_id=response.id, extra_body=agent_reference
        )
        print(f"Response (status {response.status}):")
        approvals, more_calls = show_output(response, totals)
        calls += more_calls
    print(f"\nAgent: {response.output_text}")
    return response, calls


def called(calls, name, **expected):
    """True if one of the calls is `name` with all the expected arguments."""
    for call in calls:
        arguments = json.loads(call.arguments or "{}")
        if call.name == name and all(arguments.get(key) == value for key, value in expected.items()):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Foundry agent using the Azure Functions MCP workout tools")
    parser.add_argument("--model", default=os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini")
    return parser.parse_args()


def load_settings(model):
    server_url = os.environ["AZURE_FUNCTIONS_MCP_URL"]
    project_endpoint = get_project_endpoint()
    print(f"Project endpoint: {project_endpoint}")
    print(f"Model deployment: {model}")
    print(f"MCP server: {SERVER_LABEL} at {server_url}, allowed tools {ALLOWED_TOOLS}")
    return server_url, project_endpoint


def create_agent(project, model, server_url):
    """Create an agent version with the MCP tool; the function key comes from the
    project connection."""
    connection = project.connections.get(CONNECTION_NAME)
    print(f"Key from project connection: {connection.name} ({connection.type})")
    tool = MCPTool(
        server_label=SERVER_LABEL,
        server_url=server_url,
        allowed_tools=ALLOWED_TOOLS,
        require_approval="always",
        project_connection_id=connection.id,
    )
    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(model=model, instructions=INSTRUCTIONS, tools=[tool]),
    )
    print(f"\nAgent created: name={agent.name} version={agent.version}")
    return agent


def run_conversation(openai_client, agent):
    """Ask the questions as one conversation; return the (answer, successful calls) of
    each turn and the tokens."""
    agent_reference = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
    totals = {"input": 0, "output": 0}
    turns = []
    previous_response_id = None
    for question in QUESTIONS:
        response, calls = ask(openai_client, agent_reference, question, previous_response_id, totals)
        turns.append((response.output_text or "", calls))
        previous_response_id = response.id
    return turns, totals


def cleanup(project, agent):
    project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
    print(f"\nAgent version deleted: {agent.name} v{agent.version}")


def check_results(turns):
    """Turn 1 listed the goals, turn 2 planned a 30-minute strength workout, and both
    turns answered."""
    checks = []
    if len(turns) == len(QUESTIONS):
        (answer_1, calls_1), (answer_2, calls_2) = turns
        checks = [
            ("turn 1 called list_workout_goals", called(calls_1, "list_workout_goals")),
            (
                "turn 2 called suggest_workout(strength, 30)",
                called(calls_2, "suggest_workout", goal="strength", minutes=30),
            ),
            ("both turns answered", bool(answer_1.strip()) and bool(answer_2.strip())),
        ]
    print("Checks:")
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return bool(checks) and all(passed for _, passed in checks)


def main():
    load_env()
    args = parse_args()
    server_url, project_endpoint = load_settings(args.model)
    if not check_function_app(server_url):
        raise SystemExit(1)

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project:
        agent = create_agent(project, args.model, server_url)
        openai_client = project.get_openai_client()
        try:
            turns, totals = run_conversation(openai_client, agent)
        finally:
            cleanup(project, agent)

    print(f"Total tokens: input {totals['input']}, output {totals['output']}")
    passed = check_results(turns)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
