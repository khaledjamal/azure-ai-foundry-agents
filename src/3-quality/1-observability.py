"""Observability lab (1-Observability.ipynb) as a script: tracing a chat call and a
file-search agent with OpenTelemetry, to the console or to Application Insights (shown
on the Foundry portal's Tracing page).

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/3-quality_attributes/1-Observability.ipynb

This script makes an AI app's behavior visible with tracing: every model and agent call
becomes a span, a timed record with its model, token counts and status. This is how a
team finds a slow step, a failed run or a jump in token use in production. Prompts and
answers stay out of the traces unless asked for, since they may hold private data.

What it does:
1. Lists the project's connections.
2. Makes one chat call.
3. Builds a small file-search agent over two small Markdown files and asks it three
   questions on one thread.
4. Deletes everything it created.

Tracing setup (replaces the notebook's project_client.telemetry.enable() and .inference,
tracing calls that no longer exist):
- OpenAIInstrumentor (opentelemetry-instrumentation-openai-v2) traces the OpenAI client
  calls.
- AIAgentsInstrumentor (azure-ai-agents) traces agent, thread, message and run calls.
- --destination console: spans go to a local OpenTelemetry tracer, printed one line
  each.
- --destination azure-monitor: spans also go to the Application Insights resource
  connected to the project
  (project_client.telemetry.get_application_insights_connection_string()), via
  configure_azure_monitor; there they also appear on the Foundry portal's Tracing page.
- --record-content also records prompts and answers in the spans (off by default).

Run from the project root:
    uv run python src/3-quality/1-observability.py --destination console
    uv run python src/3-quality/1-observability.py --destination azure-monitor
        --record-content
"""

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

from azure.ai.agents.models import FilePurpose, FileSearchTool, MessageRole, RunStatus
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from workshop_utils import http_log
from workshop_utils.settings import load_env

SERVICE_NAME = "observability-lab"
AGENT_NAME = "health-search-agent"
VECTOR_STORE_NAME = "health_resources_example"

AGENT_INSTRUCTIONS = """
You are a health resource advisor with access to dietary and recipe files.
You:
1. Always present disclaimers (you're not a medical professional)
2. Provide references to files when possible
3. Focus on general nutrition or recipe tips.
4. Encourage professional consultation for more detailed advice.
""".strip()

SAMPLE_FILES = {
    "recipes.md": """# Healthy Recipes Database

## Gluten-Free Recipes
1. Quinoa Bowl
   - Ingredients: quinoa, vegetables, olive oil
   - Instructions: Cook quinoa, add vegetables

2. Rice Pasta
   - Ingredients: rice pasta, mixed vegetables
   - Instructions: Boil pasta, sauté vegetables

## Diabetic-Friendly Recipes
1. Low-Carb Stir Fry
   - Ingredients: chicken, vegetables, tamari sauce
   - Instructions: Cook chicken, add vegetables

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

CHAT_QUESTION = "What's a simple 5-minute warmup routine?"
AGENT_QUESTIONS = [
    "Could you suggest a gluten-free lunch recipe?",
    "Show me some heart-healthy meal ideas.",
    "What guidelines do you have for someone with diabetes?",
]

SHOWN_ATTRIBUTES = {
    "gen_ai.request.model": "model",
    "gen_ai.usage.input_tokens": "in",
    "gen_ai.usage.output_tokens": "out",
    "gen_ai.agent.name": "agent",
    "gen_ai.thread.run.status": "status",
}


class CompactConsoleExporter(SpanExporter):
    """Prints one line per GenAI or lab span; Azure SDK and HTTP spans are only
    counted."""

    def __init__(self):
        self.hidden = 0

    def export(self, spans):
        for span in spans:
            attributes = span.attributes or {}
            if span.instrumentation_scope.name != SERVICE_NAME and not any(k.startswith("gen_ai.") for k in attributes):
                self.hidden += 1
                continue
            details = "  ".join(
                f"{label}={attributes[key]}" for key, label in SHOWN_ATTRIBUTES.items() if key in attributes
            )
            duration_ms = (span.end_time - span.start_time) / 1e6
            print(f"  [span] {span.name:<45} {duration_ms:7.0f} ms  {details}")
        return SpanExportResult.SUCCESS


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


def setup_tracing(destination, project_client, record_content):
    # Content recording must be set before the instrumentors run.
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_ONLY" if record_content else "NO_CONTENT"
    os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true" if record_content else "false"
    resource = Resource.create({"service.name": SERVICE_NAME})

    if destination == "azure-monitor":
        from azure.monitor.opentelemetry import configure_azure_monitor

        connection_string = project_client.telemetry.get_application_insights_connection_string()
        # sampling_ratio=1.0 keeps every span; the default rate-limited sampler drops
        # spans in bursts.
        configure_azure_monitor(connection_string=connection_string, resource=resource, sampling_ratio=1.0)
        provider = trace.get_tracer_provider()
        print("Tracing to Application Insights (connection found on the Foundry project) and the console.")
    else:
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        print("Tracing to the console only.")
    exporter = CompactConsoleExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    from azure.ai.agents.telemetry import AIAgentsInstrumentor
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()
    AIAgentsInstrumentor().instrument()
    print(f"Prompt and answer content in spans: {'recorded' if record_content else 'not recorded'}")
    print("Spans are printed when they end, so inner steps appear before the step that contains them.")
    return provider, exporter


def list_connections(project_client):
    for connection in project_client.connections.list():
        print(f"  {connection.name}: {getattr(connection.type, 'value', connection.type)}")


def chat_call(openai_client, model):
    print(f"User: {CHAT_QUESTION}")
    response = openai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": CHAT_QUESTION}])
    print(f"Assistant: {response.choices[0].message.content}")


def run_agent(project_client, model, tracer):
    uploaded, vector_store, agent, thread = [], None, None, None
    failed_runs = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, content in SAMPLE_FILES.items():
                path = Path(temp_dir) / name
                path.write_text(content, encoding="utf-8")
                uploaded.append(
                    project_client.agents.files.upload_and_poll(file_path=str(path), purpose=FilePurpose.AGENTS).id
                )
        vector_store = project_client.agents.vector_stores.create_and_poll(file_ids=uploaded, name=VECTOR_STORE_NAME)
        file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
        agent = project_client.agents.create_agent(
            model=model,
            name=AGENT_NAME,
            instructions=AGENT_INSTRUCTIONS,
            tools=file_search_tool.definitions,
            tool_resources=file_search_tool.resources,
        )
        thread = project_client.agents.threads.create()
        print(f"Agent {agent.id}, vector store {vector_store.id}, thread {thread.id}")

        for question in AGENT_QUESTIONS:
            with tracer.start_as_current_span("ask-agent") as span:
                span.set_attribute("lab.question", question)
                print(f"\nUser: {question}")
                project_client.agents.messages.create(thread_id=thread.id, role="user", content=question)
                run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
                print(f"Run status: {run.status}")
                if run.status != RunStatus.COMPLETED:
                    failed_runs.append((question, run.status, run.last_error))
                answer = project_client.agents.messages.get_last_message_text_by_role(
                    thread_id=thread.id, role=MessageRole.AGENT
                )
                if answer:
                    print(f"Assistant: {answer.text.value}")
    finally:
        print("\nCleanup:")
        if thread:
            project_client.agents.threads.delete(thread.id)
            print(f"  deleted thread {thread.id}")
        if agent:
            project_client.agents.delete_agent(agent.id)
            print(f"  deleted agent {agent.id}")
        if vector_store:
            project_client.agents.vector_stores.delete(vector_store.id)
            print(f"  deleted vector store {vector_store.id}")
        for file_id in uploaded:
            project_client.agents.files.delete(file_id)
            print(f"  deleted file {file_id}")
    return failed_runs


def main():
    parser = argparse.ArgumentParser(description="Observability lab")
    parser.add_argument("--destination", choices=["console", "azure-monitor"], default="console")
    parser.add_argument("--record-content", action="store_true", help="record prompts and answers in the spans")
    args = parser.parse_args()

    print_section(f"OBSERVABILITY LAB - destination: {args.destination}")
    load_environment()
    model = require_env("MODEL_DEPLOYMENT_NAME")
    api_version = require_env("OPENAI_API_VERSION")
    project_endpoint = get_project_endpoint()
    print(f"Project endpoint: {project_endpoint}")
    print(f"Model deployment: {model}")

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client(api_version=api_version) as openai_client,
    ):
        provider, exporter = setup_tracing(args.destination, project_client, args.record_content)
        tracer = trace.get_tracer(SERVICE_NAME)
        session_id = uuid.uuid4().hex[:12]

        with tracer.start_as_current_span("observability-lab") as root:
            root.set_attribute("lab.session_id", session_id)
            trace_id = format(root.get_span_context().trace_id, "032x")
            print(f"Trace ID: {trace_id}  (session {session_id})")

            print_section("1. Project connections")
            list_connections(project_client)

            print_section("2. Chat call")
            with tracer.start_as_current_span("chat-call"):
                chat_call(openai_client, model)

            print_section("3. File-search agent")
            failed_runs = run_agent(project_client, model, tracer)

        provider.force_flush()

    print_section("Result")
    print(f"Spans not printed (Azure SDK and HTTP calls): {exporter.hidden}")
    if args.destination == "azure-monitor":
        print(
            f"In the Foundry portal: project -> Tracing; or in Application Insights: search for operation ID {trace_id}."
        )
        print("Traces usually appear there within a few minutes.")
    if failed_runs:
        print(f"FINISHED WITH ERRORS: {len(failed_runs)} agent run(s) did not complete")
        for question, status, error in failed_runs:
            print(f"  {question!r}: status {status}, error: {error}")
        sys.exit(1)
    print(f"COMPLETED: chat call and {len(AGENT_QUESTIONS)} agent runs finished; trace {trace_id}")


if __name__ == "__main__":
    http_log.enable()
    main()
