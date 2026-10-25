"""Code interpreter lab (2-code_interpreter.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/2-code_interpreter.ipynb

This script gives an agent the Code Interpreter tool: a sandbox where the agent writes and runs its own Python code.
This is how an agent computes numbers instead of guessing them.

What it does:
1. Uploads a small nutrition table (CSV).
2. Creates a health calculator agent with the Code Interpreter tool.
3. Asks it to calculate and explain a BMI, and to analyze the nutrition data with a chart.
4. Saves any charts it draws.
5. Deletes everything it created in the project: threads, the agent, and the uploaded and generated files.

Connection method: AIProjectClient from azure-ai-projects, created with the Foundry project
endpoint and an Entra ID token from the Azure CLI login.

Run from the project root:
    uv run python src/2-agents/2-code-interpreter.py [--output-dir DIR]
"""

import argparse
import csv
import os
import tempfile
from pathlib import Path

from azure.ai.agents.models import CodeInterpreterTool, FilePurpose, ListSortOrder, MessageRole
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

AGENT_NAME = "health-calculator-agent"

AGENT_INSTRUCTIONS = """
You are a health calculator agent that can:
1. Calculate and interpret BMI
2. Analyze provided nutrition data
3. Generate charts/plots
4. Include disclaimers that you are not a medical professional

When you draw a chart, save it as a PNG file in /mnt/data/ with plt.savefig() and give the user the file;
plt.show() does not return anything to the user.
""".strip()

NUTRITION_ROWS = [
    ["Date", "Calories", "Protein_g", "Carbs_g", "Fat_g", "Fiber_g"],
    ["2024-01-01", 2100, 80, 250, 70, 25],
    ["2024-01-02", 1950, 75, 230, 65, 22],
    ["2024-01-03", 2300, 85, 270, 75, 28],
    ["2024-01-04", 2050, 78, 245, 68, 24],
    ["2024-01-05", 1900, 72, 225, 63, 21],
    ["2024-01-06", 2200, 82, 260, 73, 26],
    ["2024-01-07", 2150, 79, 255, 71, 23],
]

BMI_REQUEST = (
    "Calculate BMI for \n"
    "Height: 70 inches\n"
    "Weight: 180 pounds\n"
    "Please: \n"
    "1. Show calculation \n"
    "2. Interpret the result \n"
    "3. Include disclaimers \n"
)

NUTRITION_REQUEST = (
    "Analyze the CSV file with daily nutrition data.\n"
    "1. Compute average daily macros (calories, protein, carbs, fat, fiber).\n"
    "2. Create a chart to show trends and save it as /mnt/data/nutrition_trends.png. Use two panels, one above the\n"
    "   other: daily calories in the top panel, and protein, carbs, fat and fiber in grams in the bottom panel, so\n"
    "   the gram values are not flattened by the calorie scale.\n"
    "3. Discuss any insights or disclaimers.\n"
)

# (section title, request); each request gets its own thread.
REQUESTS = [("BMI calculation", BMI_REQUEST), ("Nutrition analysis", NUTRITION_REQUEST)]


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


def write_sample_csv(folder):
    path = Path(folder) / "nutrition_data.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(NUTRITION_ROWS)
    print(f"Created sample data file: {path.name} ({len(NUTRITION_ROWS) - 1} rows)")
    return path


def ask(project_client, agent_id, request):
    thread = project_client.agents.threads.create()
    print(f"Created thread: {thread.id}")
    print(f"User: {request.strip()}")
    project_client.agents.messages.create(thread_id=thread.id, role="user", content=request)
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)
    print(f"Run status: {run.status}")
    if run.usage:
        print(f"(tokens: prompt {run.usage.prompt_tokens}, completion {run.usage.completion_tokens})")
    if run.last_error:
        print(f"Run error: {run.last_error}")
    return thread, run


def code_interpreter_images(project_client, thread_id, run_id):
    # Charts can show up in the run's tool-call outputs as well as in the messages.
    file_ids = []
    steps = project_client.agents.run_steps.list(thread_id=thread_id, run_id=run_id, order=ListSortOrder.ASCENDING)
    for step in steps:
        for call in getattr(step.step_details, "tool_calls", None) or []:
            details = getattr(call, "code_interpreter", None)
            if not details:
                continue
            outputs = details.outputs or []
            print(
                f"Code interpreter call: {len(details.input.splitlines())} lines of code, "
                f"outputs {[output.type for output in outputs]}"
            )
            print("  " + details.input.replace("\n", "\n  "))
            for output in outputs:
                image = getattr(output, "image", None)
                if image:
                    file_ids.append(image.file_id)
    return file_ids


def show_results(project_client, thread_id, run_id, output_dir, generated_file_ids):
    image_ids = code_interpreter_images(project_client, thread_id, run_id)
    messages = project_client.agents.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
    for message in messages:
        if message.role != MessageRole.AGENT:
            continue
        print(f"\n[message content types: {[content.type for content in message.content]}]")
        for text_message in message.text_messages:
            print(f"ASSISTANT: {text_message.text.value}")
        image_ids += [image.image_file.file_id for image in message.image_contents]
        # A file saved under /mnt/data comes back as a link in the text (sandbox:/mnt/data/...).
        for annotation in message.file_path_annotations:
            print(f"[file link: {annotation.text} -> {annotation.file_path.file_id}]")
            image_ids.append(annotation.file_path.file_id)

    for file_id in dict.fromkeys(image_ids):
        file_name = f"chart_{file_id}.png"
        project_client.agents.files.save(file_id=file_id, file_name=file_name, target_dir=str(output_dir))
        generated_file_ids.append(file_id)
        print(f"\nSaved chart: {Path(output_dir) / file_name}")
    if not image_ids:
        print("\nNo chart image was returned for this request.")


def parse_args():
    parser = argparse.ArgumentParser(description="Code interpreter lab")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="folder for charts the agent draws (default: current folder)"
    )
    return parser.parse_args()


def load_settings():
    model = require_env("MODEL_DEPLOYMENT_NAME")
    project_endpoint = get_project_endpoint()
    print("Connection method: AIProjectClient (project_client.agents)")
    print(f"Project endpoint:  {project_endpoint}")
    print(f"Model deployment:  {model}")
    return model, project_endpoint


def upload_sample_data(project_client):
    """Step 1: write the nutrition table to a temporary CSV file and upload it to the project; return the file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = write_sample_csv(temp_dir)
        uploaded_file = project_client.agents.files.upload_and_poll(
            file_path=str(csv_path),
            purpose=FilePurpose.AGENTS,
        )
    print(f"Uploaded CSV file: {uploaded_file.id}")
    return uploaded_file


def create_agent(project_client, model, uploaded_file):
    """Step 2: create the agent with the Code Interpreter tool, which can read the uploaded file."""
    code_tool = CodeInterpreterTool(file_ids=[uploaded_file.id])
    agent = project_client.agents.create_agent(
        model=model,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=code_tool.definitions,
        tool_resources=code_tool.resources,
    )
    print(f"Created agent: {agent.name} ({agent.id})")
    return agent


def cleanup(project_client, threads, agent, uploaded_file, generated_file_ids):
    """Step 5: delete what the run created: threads, the agent, and the uploaded and generated files."""
    for thread in threads:
        project_client.agents.threads.delete(thread.id)
        print(f"Deleted thread: {thread.id}")
    if agent:
        project_client.agents.delete_agent(agent.id)
        print(f"Deleted agent: {agent.id}")
    for file_id in ([uploaded_file.id] if uploaded_file else []) + generated_file_ids:
        project_client.agents.files.delete(file_id)
        print(f"Deleted file from the project: {file_id}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_section("CODE INTERPRETER AGENT - AIProjectClient")
    load_environment()
    model, project_endpoint = load_settings()

    with AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) as project_client:
        uploaded_file = agent = None
        threads = []
        generated_file_ids = []
        try:
            print_section("1. Upload sample data and create the agent")
            uploaded_file = upload_sample_data(project_client)
            agent = create_agent(project_client, model, uploaded_file)

            # Steps 3 and 4: each request on its own thread; charts are saved to the output folder.
            for number, (title, request) in enumerate(REQUESTS, start=2):
                print_section(f"{number}. {title}")
                thread, run = ask(project_client, agent.id, request)
                threads.append(thread)
                show_results(project_client, thread.id, run.id, args.output_dir, generated_file_ids)
        finally:
            print_section("Cleanup")
            cleanup(project_client, threads, agent, uploaded_file, generated_file_ids)

    print_section("DONE")


if __name__ == "__main__":
    http_log.enable()
    main()
