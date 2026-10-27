"""Delete Foundry agents with an exact name from the project; all other agents are kept.

Lists every agent first (id, name, model, created time), then deletes the ones whose
name matches. With --dry-run it only lists them.

Run from the project root:
    uv run python scripts/delete-agents.py --name fun-fit-health-advisor [--dry-run]
"""

import argparse
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env


def get_project_endpoint():
    direct_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if direct_endpoint:
        return direct_endpoint.rstrip("/")
    services_endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/")
    project_name = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{services_endpoint}/api/projects/{project_name}"


def main():
    parser = argparse.ArgumentParser(description="Delete Foundry agents by exact name")
    parser.add_argument("--name", required=True, help="exact agent name to delete")
    parser.add_argument("--dry-run", action="store_true", help="list only, delete nothing")
    args = parser.parse_args()

    load_env()
    project_endpoint = get_project_endpoint()
    print(f"Project endpoint: {project_endpoint}")

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project_client:
        agents = sorted(project_client.agents.list_agents(), key=lambda a: a.created_at)
        print(f"\nAgents in the project: {len(agents)}")
        for agent in agents:
            action = "DELETE" if agent.name == args.name else "keep"
            print(
                f"  {action:6}  {agent.id}  {agent.name:25}  {agent.model:12}  created {agent.created_at.isoformat()}"
            )

        targets = [a for a in agents if a.name == args.name]
        print(f"\nMatching '{args.name}': {len(targets)}; kept: {len(agents) - len(targets)}")
        if args.dry_run:
            print("Dry run: nothing deleted.")
            return

        for agent in targets:
            project_client.agents.delete_agent(agent.id)
            print(f"Deleted agent: {agent.id}")

        remaining = list(project_client.agents.list_agents())
        print(f"\nAgents left in the project: {len(remaining)}")
        for agent in remaining:
            print(f"  {agent.id}  {agent.name}  {agent.model}  created {agent.created_at.isoformat()}")


if __name__ == "__main__":
    http_log.enable()
    main()
