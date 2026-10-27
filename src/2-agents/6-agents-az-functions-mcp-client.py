"""Direct test of the Azure Functions MCP server: list its tools and call them, without
any model or agent.

This script talks to the workout MCP server directly, with no model and no agent, so
there are no model charges. MCP (Model Context Protocol) is the open standard that lets
AI agents use outside tools. Testing the server separately helps tell server or tool
problems apart from problems in the agent integration.

What it does:
1. Lists the server's tools.
2. Calls them.

Servers:
- Local: `func start` in src/functions-mcp-server.
- Deployed on Azure Functions: set AZURE_FUNCTIONS_MCP_URL
  (https://<app>.azurewebsites.net/runtime/webhooks/mcp) and AZURE_FUNCTIONS_MCP_KEY
  (the `mcp_extension` system key) in .env; the key is sent as x-functions-key.

Needs the `mcp` package (in the separate agents venv). Run from the project root:
    source ../../.venv-agents-v2/bin/activate
    python src/2-agents/6-agents-az-functions-mcp-client.py [--url URL] [--verbose]
"""

import argparse
import asyncio
import json
import logging
import os

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from workshop_utils import http_log
from workshop_utils.settings import load_env

LOCAL_URL = "http://localhost:7071/runtime/webhooks/mcp"
EXPECTED_TOOLS = {"list_workout_goals", "suggest_workout"}


def tool_text(result):
    return "\n".join(item.text for item in result.content if getattr(item, "type", "") == "text")


async def check_server(url, key):
    headers = {"x-functions-key": key} if key else {}
    checks = []
    async with (
        create_mcp_http_client(headers=headers) as http_client,
        streamable_http_client(url, http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        info = await session.initialize()
        print(f"Server: {info.server_info.name} {info.server_info.version}")

        tools = (await session.list_tools()).tools
        print(f"\nTools: {len(tools)}")
        for tool in tools:
            print(f"  {tool.name}: {tool.description}")
            print(f"    input schema: {json.dumps(tool.input_schema)}")
        checks.append(("both tools listed", EXPECTED_TOOLS <= {tool.name for tool in tools}))

        print("\nCall list_workout_goals()")
        result = await session.call_tool("list_workout_goals", {})
        text = tool_text(result)
        print(f"  {text}")
        goals = [item["goal"] for item in json.loads(text)["goals"]] if not result.is_error else []
        checks.append(("goals returned", "cardio" in goals))

        print("\nCall suggest_workout(goal='cardio', minutes=30)")
        result = await session.call_tool("suggest_workout", {"goal": "cardio", "minutes": 30})
        text = tool_text(result)
        print(f"  {text}")
        plan = json.loads(text) if not result.is_error else {}
        parts = sum(plan.get(part, {}).get("minutes", 0) for part in ("warm_up", "main", "cool_down"))
        checks.append(("30-minute cardio plan adds up", plan.get("goal") == "cardio" and parts == 30))

        print("\nCall suggest_workout(goal='yoga', minutes=30), expecting a clear error")
        result = await session.call_tool("suggest_workout", {"goal": "yoga", "minutes": 30})
        text = tool_text(result)
        print(f"  {text}")
        checks.append(("unknown goal rejected", "error" in text))
    return checks


def main():
    load_env()
    parser = argparse.ArgumentParser(description="List and call the tools of the Functions MCP server")
    parser.add_argument("--url", default=os.getenv("AZURE_FUNCTIONS_MCP_URL") or LOCAL_URL)
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request with its status code")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="  [%(name)s] %(message)s")
    key = os.getenv("AZURE_FUNCTIONS_MCP_KEY") if args.url != LOCAL_URL else None
    print(f"MCP endpoint: {args.url} (key {'sent' if key else 'not sent'})")

    checks = asyncio.run(check_server(args.url, key))
    print("\nChecks:")
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    raise SystemExit(0 if all(passed for _, passed in checks) else 1)


if __name__ == "__main__":
    http_log.enable()
    main()
