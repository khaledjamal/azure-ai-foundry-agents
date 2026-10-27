"""Send raw MCP JSON-RPC requests over HTTP and print each status, header and body, to
debug an MCP endpoint.

Sends initialize, the initialized notification, then tools/list, reusing the
Mcp-Session-Id the server returns. URL and key come from AZURE_FUNCTIONS_MCP_URL and
AZURE_FUNCTIONS_MCP_KEY in .env (the key is sent as x-functions-key and never printed).
Run from the repo root:
    python scripts/probe-mcp-http.py [--url URL]
"""

import argparse
import json
import os

import requests

from workshop_utils import http_log
from workshop_utils.settings import load_env

SHOWN_HEADERS = ("content-type", "mcp-session-id", "www-authenticate")


def post(client, url, headers, message):
    print(f"\n>>> {message['method']}")
    response = client.post(url, headers=headers, json=message, timeout=60)
    print(f"<<< HTTP {response.status_code}")
    for name in SHOWN_HEADERS:
        if name in response.headers:
            print(f"    {name}: {response.headers[name]}")
    print(f"    body: {response.text[:2000]}")
    return response


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Raw MCP requests to an HTTP endpoint")
    parser.add_argument("--url", default=os.getenv("AZURE_FUNCTIONS_MCP_URL"))
    args = parser.parse_args()
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if key := os.getenv("AZURE_FUNCTIONS_MCP_KEY"):
        headers["x-functions-key"] = key
    print(f"MCP endpoint: {args.url} (key {'sent' if key else 'not sent'})")

    with requests.Session() as client:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "probe", "version": "1.0"},
            },
        }
        response = post(client, args.url, headers, init)
        if session_id := response.headers.get("mcp-session-id"):
            headers["Mcp-Session-Id"] = session_id
        post(client, args.url, headers, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        post(client, args.url, headers, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        call = {"name": "list_workout_goals", "arguments": {}}
        post(client, args.url, headers, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": call})
    print(json.dumps({"done": True}))


if __name__ == "__main__":
    http_log.enable()
    main()
