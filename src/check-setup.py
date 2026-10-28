"""Read-only setup check for the labs, the script version of the three 1-introduction
notebooks.

Workshop notebooks:
https://github.com/Azure/ai-foundry-workshop/tree/273c24c/1-introduction

This script checks that everything the labs need is in place before any of them runs. It
prints a result for each check, so a setup problem shows up here with a clear message
instead of as an error in the middle of a lab. It costs nothing because it calls no
model.

The notebooks troubleshoot DefaultAzureCredential, create the project client and list
models and connections. This script does the same checks in one go:
  1. which lab settings are set in .env (names only, values are never printed),
  2. Entra ID sign-in works: a token for Azure AI services from DefaultAzureCredential
     (`az login` locally),
  3. the Foundry project answers: its model deployments and connections, through
     AIProjectClient,
  4. the project has what the labs expect: every *_DEPLOYMENT_NAME setting names an
     existing deployment and the Bing connection exists,
  5. Azure AI Search accepts its key: the index names can be listed.
Exits with 1 if sign-in, the project or a required setting fails; optional labs only get
a warning.

Run from the project root:
    uv run python src/check-setup.py
"""

import os
from datetime import UTC, datetime

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient

from workshop_utils import http_log
from workshop_utils.settings import ENV_FILE, load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
# Settings per lab group; "required" groups fail the check when a setting is missing,
# the others only warn.
SETTINGS = [
    ("chat, agents (required)", True, ["AZURE_AI_SERVICES_ENDPOINT", "OPENAI_ENDPOINT", "OPENAI_API_VERSION"]),
    (
        "embeddings, RAG, AI Search agent",
        False,
        ["EMBEDDING_MODEL_DEPLOYMENT_NAME", "AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_API_KEY"],
    ),
    ("evaluation judge model", False, ["OPENAI_DEPLOYMENT_NAME"]),
    ("Grounding with Bing", False, ["GROUNDING_WITH_BING_CONNECTION_NAME"]),
    ("Azure Functions MCP", False, ["AZURE_FUNCTIONS_MCP_URL", "AZURE_FUNCTIONS_MCP_KEY"]),
    ("image scripts", False, ["IMAGE_EMBEDDING_MODEL_DEPLOYMENT_NAME", "IMAGE_MODEL_DEPLOYMENT_NAME"]),
    ("Phi-4", False, ["PHI_MODEL_DEPLOYMENT_NAME", "PHI_REASONING_MODEL_DEPLOYMENT_NAME"]),
    ("DeepSeek", False, ["DEEPSEEK_MODEL_DEPLOYMENT_NAME"]),
]


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def get_project_endpoint():
    direct_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if direct_endpoint:
        return direct_endpoint.rstrip("/")
    services_endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/")
    project_name = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{services_endpoint}/api/projects/{project_name}"


def check_settings(checks):
    """Step 1: which lab settings are set (names only). Stops with the summary if the
    Foundry project isn't set."""
    print_section("1. Settings in .env")
    if not ENV_FILE.exists():
        print(f"Missing: {ENV_FILE} (copy .env.example)")
        raise SystemExit(1)
    load_env()
    for group, required, names in SETTINGS:
        missing = [name for name in names if not os.getenv(name)]
        status = "PASS" if not missing else ("FAIL" if required else "WARN")
        print(f"  {status}  {group}: " + (f"missing {', '.join(missing)}" if missing else "all set"))
        checks.append((status, f"settings for {group}"))
    project_set = bool(os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("AZURE_AI_PROJECT_NAME"))
    print(
        f"  {'PASS' if project_set else 'FAIL'}  Foundry project: "
        + ("set" if project_set else "missing AZURE_AI_PROJECT_NAME (or AZURE_AI_PROJECT_ENDPOINT)")
    )
    checks.append(("PASS" if project_set else "FAIL", "Foundry project setting"))
    if not project_set:
        report(checks)


def check_sign_in(checks, credential):
    """Step 2: get a token for Azure AI services. Stops with the summary if sign-in
    fails."""
    print_section("2. Sign-in (DefaultAzureCredential)")
    try:
        token = credential.get_token(COGNITIVE_SERVICES_SCOPE)
        expires = datetime.fromtimestamp(token.expires_on, tz=UTC)
        print(f"  Token for Azure AI services received, valid until {expires:%Y-%m-%d %H:%M} UTC")
        checks.append(("PASS", "sign-in"))
    except Exception as exc:
        print(f"  Sign-in failed: {exc}\n  Run `az login` and try again.")
        checks.append(("FAIL", "sign-in"))
        report(checks)


def check_project(checks, credential):
    """Step 3: list the project's deployments and connections; return both. Stops with
    the summary on an error."""
    print_section("3. Foundry project")
    endpoint = get_project_endpoint()
    print(f"  Project endpoint: {endpoint}")
    try:
        with AIProjectClient(endpoint=endpoint, credential=credential) as project_client:
            deployments = list(project_client.deployments.list())
            connections = list(project_client.connections.list())
    except Exception as exc:
        print(f"  Could not read the project: {exc}")
        checks.append(("FAIL", "project access"))
        report(checks)
    checks.append(("PASS", "project access"))
    print(f"  Deployments ({len(deployments)}):")
    for deployment in deployments:
        sku = getattr(deployment.sku, "name", "") if getattr(deployment, "sku", None) else ""
        print(
            f"    {deployment.name:26} {deployment.model_publisher or '':10} {deployment.model_name} "
            f"{deployment.model_version or ''} {sku}"
        )
    print(f"  Connections ({len(connections)}):")
    for connection in connections:
        print(f"    {connection.name:40} {connection.type}")
    return deployments, connections


def check_named_deployments(checks, deployments, connections):
    """Step 4: every *_DEPLOYMENT_NAME setting names an existing deployment, and the
    Bing connection exists."""
    print_section("4. Settings that name deployments and connections")
    model = os.getenv("MODEL_DEPLOYMENT_NAME") or "gpt-4o-mini"
    deployment_names = {deployment.name for deployment in deployments}
    lab_names = [name for _, _, names in SETTINGS for name in names if name.endswith("_DEPLOYMENT_NAME")]
    named = {"MODEL_DEPLOYMENT_NAME": model}
    named.update({name: os.environ[name] for name in lab_names if os.getenv(name)})
    for name, value in sorted(named.items()):
        found = value in deployment_names
        status = "PASS" if found else ("FAIL" if name == "MODEL_DEPLOYMENT_NAME" else "WARN")
        print(f"  {status}  {name}={value}" + ("" if found else " (no deployment with that name)"))
        checks.append((status, f"deployment for {name}"))
    bing = os.getenv("GROUNDING_WITH_BING_CONNECTION_NAME")
    if bing:
        found = bing in {connection.name for connection in connections}
        print(f"  {'PASS' if found else 'WARN'}  GROUNDING_WITH_BING_CONNECTION_NAME names a project connection")
        checks.append(("PASS" if found else "WARN", "Bing connection"))


def check_search(checks):
    """Step 5: Azure AI Search accepts its key, shown by listing the index names.
    Skipped if not set."""
    print_section("5. Azure AI Search")
    search_endpoint, search_key = os.getenv("AZURE_AI_SEARCH_ENDPOINT"), os.getenv("AZURE_AI_SEARCH_API_KEY")
    if search_endpoint and search_key:
        try:
            names = list(SearchIndexClient(search_endpoint, AzureKeyCredential(search_key)).list_index_names())
            print(f"  Key works; {len(names)} index(es): {', '.join(names) or '(none)'}")
            checks.append(("PASS", "Azure AI Search"))
        except Exception as exc:
            print(f"  Could not list indexes: {exc}")
            checks.append(("WARN", "Azure AI Search"))
    else:
        print("  Skipped (AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_API_KEY not set)")


def main():
    checks = []  # (status, message) with status PASS, WARN or FAIL
    check_settings(checks)
    # Made after check_settings(), which loads .env: the credential reads its settings
    # from the environment when it is created.
    with DefaultAzureCredential() as credential:
        check_sign_in(checks, credential)
        deployments, connections = check_project(checks, credential)
    check_named_deployments(checks, deployments, connections)
    check_search(checks)
    report(checks)


def report(checks):
    print_section("Summary")
    for status in ("FAIL", "WARN"):
        for check_status, message in checks:
            if check_status == status:
                print(f"  {status}  {message}")
    failed = any(status == "FAIL" for status, _ in checks)
    passed = sum(1 for status, _ in checks if status == "PASS")
    print(f"  {passed} passed, {sum(1 for s, _ in checks if s == 'WARN')} warnings, {'FAIL' if failed else 'OK'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    http_log.enable()
    main()
