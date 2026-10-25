"""Evaluation lab (2-evaluation.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/3-quality_attributes/2-evaluation.ipynb

This script measures answer quality automatically, the way a team checks a model or prompt change before release. Its
three sample health & fitness answers are poor on purpose (e.g. "London" as the capital of France), so low quality
scores are the expected result and show that the evaluators work.

It scores each answer in four ways, with azure-ai-evaluation:
- F1 score: word overlap between the answer and the reference answer (no model call)
- Relevance and Groundedness: a second model, gpt-4o, acts as the judge (AI-assisted)
- Violence: Azure AI Foundry's safety evaluation service checks for violent content
- Response length: a custom code-based evaluator

Results and connection:
- The results are also logged to the Foundry project, where they show up on the Evaluations page. This replaces the
  notebook's cloud evaluation section, which used project_client.evaluations from the beta SDK; that API is not in
  azure-ai-projects 1.0.0.
- The judge model uses the Azure OpenAI endpoint (OPENAI_ENDPOINT); the Violence evaluator and result logging use the
  Foundry project endpoint; all with an Entra ID token from the Azure CLI login.

Run from the project root:
    uv run python src/3-quality/2-evaluation.py [--output-dir DIR]
"""

import argparse
import json
import os
from pathlib import Path

from azure.ai.evaluation import (
    F1ScoreEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    ViolenceEvaluator,
    evaluate,
)
from azure.identity import DefaultAzureCredential

from workshop_utils import http_log
from workshop_utils.settings import load_env

SAMPLE_DATA = [
    {
        "query": "How can I start a beginner workout routine at home?",
        "context": "Workout routines can include push-ups, bodyweight squats, lunges, and planks.",
        "response": "You can just go for 10 push-ups total.",
        "ground_truth": "At home, you can start with short, low-intensity workouts: push-ups, lunges, planks.",
    },
    {
        "query": "Are diet sodas healthy for daily consumption?",
        "context": "Sugar-free or diet drinks may reduce sugar intake, but they still contain artificial sweeteners.",
        "response": "Yes, diet sodas are 100% healthy.",
        "ground_truth": "Diet sodas have fewer sugars than regular soda, but 'healthy' is not guaranteed due to artificial additives.",
    },
    {
        "query": "What's the capital of France?",
        "context": "France is in Europe. Paris is the capital.",
        "response": "London.",
        "ground_truth": "Paris.",
    },
]
# Which data columns each evaluator gets as its inputs.
EVALUATOR_CONFIG = {
    "f1_score": {"column_mapping": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"}},
    "relevance": {"column_mapping": {"query": "${data.query}", "response": "${data.response}"}},
    "groundedness": {"column_mapping": {"context": "${data.context}", "response": "${data.response}"}},
    "violence": {"column_mapping": {"query": "${data.query}", "response": "${data.response}"}},
    "resp_len": {"column_mapping": {"response": "${data.response}"}},
}


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


def response_length_eval(response, **kwargs):
    """Custom code-based evaluator: length of the response in characters."""
    return {"resp_length": len(response)}


def write_sample_data(output_dir):
    path = output_dir / "health_fitness_eval_data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in SAMPLE_DATA:
            f.write(json.dumps(row) + "\n")
    print(f"Sample evaluation data: {path} ({len(SAMPLE_DATA)} rows)")
    return path


def print_rows(rows):
    for number, row in enumerate(rows, start=1):
        print_section(f"Row {number}")
        print(f"Query:        {row.get('inputs.query')}")
        print(f"Response:     {row.get('inputs.response')}")
        print(f"Ground truth: {row.get('inputs.ground_truth')}")
        for key, value in row.items():
            if key.startswith("outputs."):
                print(f"  {key[len('outputs.') :]}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation lab")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="folder for the sample data and results JSON"
    )
    return parser.parse_args()


def load_settings():
    """Return the project endpoint and the judge model's config (the Azure OpenAI deployment that scores answers)."""
    project_endpoint = get_project_endpoint()
    model_config = {
        "azure_endpoint": require_env("OPENAI_ENDPOINT"),
        "azure_deployment": require_env("OPENAI_DEPLOYMENT_NAME"),
        "api_version": require_env("OPENAI_API_VERSION"),
    }
    print(f"Judge model:       {model_config['azure_deployment']} at {model_config['azure_endpoint']}")
    print(f"Project endpoint:  {project_endpoint}")
    return project_endpoint, model_config


def build_evaluators(model_config, project_endpoint):
    """The four kinds of scoring: F1 (no model), judge model, safety service and a custom code evaluator."""
    credential = DefaultAzureCredential()
    return {
        "f1_score": F1ScoreEvaluator(),
        "relevance": RelevanceEvaluator(model_config=model_config, credential=credential),
        "groundedness": GroundednessEvaluator(model_config=model_config, credential=credential),
        "violence": ViolenceEvaluator(credential=credential, azure_ai_project=project_endpoint),
        "resp_len": response_length_eval,
    }


def run_evaluation(data_path, evaluators, project_endpoint, output_dir):
    """Score every row with every evaluator, log the results to the Foundry project and save them as JSON."""
    results_path = output_dir / "evaluation_results.json"
    result = evaluate(
        data=str(data_path),
        evaluators=evaluators,
        evaluator_config=EVALUATOR_CONFIG,
        evaluation_name="health-fitness-evaluation",
        azure_ai_project=project_endpoint,
        output_path=str(results_path),
    )
    print(f"Full results saved to: {results_path}")
    return result


def print_metrics(result):
    for name, value in sorted(result.get("metrics", {}).items()):
        print(f"  {name}: {value}")


def print_portal_link(result):
    studio_url = result.get("studio_url")
    if studio_url:
        print(f"Evaluation in the portal: {studio_url}")
    else:
        print("No portal link returned (results not logged).")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_section("EVALUATION - azure-ai-evaluation")
    load_environment()
    project_endpoint, model_config = load_settings()
    data_path = write_sample_data(args.output_dir)
    evaluators = build_evaluators(model_config, project_endpoint)

    print_section("Running evaluate() and logging results to the Foundry project")
    result = run_evaluation(data_path, evaluators, project_endpoint, args.output_dir)
    print_rows(result.get("rows", []))

    print_section("Aggregate metrics")
    print_metrics(result)

    print_section("Foundry project")
    print_portal_link(result)

    print_section("DONE")


if __name__ == "__main__":
    http_log.enable()
    main()
