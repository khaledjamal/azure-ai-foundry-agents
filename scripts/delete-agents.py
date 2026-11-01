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

# list_agents() fetches its next pages by itself, so no print can go right before them;
# http_log prints the description before a matching call that it logs in full.
CALL_DESCRIPTIONS = {
    "GET */assistants*after=*": "list_agents(): ask Foundry Agent Service for the next page of agents.",
}


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
        print("\n>>> List the agents in Foundry Agent Service, one call per page.")
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
            print(f"\n>>> Delete the agent {agent.id} from Foundry Agent Service.")
            project_client.agents.delete_agent(agent.id)
            print(f"Deleted agent: {agent.id}")

        print("\n>>> List the agents left in Foundry Agent Service, one call per page.")
        remaining = list(project_client.agents.list_agents())
        print(f"\nAgents left in the project: {len(remaining)}")
        for agent in remaining:
            print(f"  {agent.id}  {agent.name}  {agent.model}  created {agent.created_at.isoformat()}")


if __name__ == "__main__":
    http_log.enable(call_descriptions=CALL_DESCRIPTIONS)
    main()
