"""Remote MCP server on Azure Functions with two workout tools (fixed logic, no model).

This file is the server side of the MCP lab: a small app on Azure Functions that offers
two workout tools to any AI agent that speaks MCP (Model Context Protocol), the open
standard for giving agents tools.

Tools (the plans come from fixed rules, not a model, so the answers are predictable,
with no model charges):
- One lists the workout goals it knows.
- The other builds a workout plan for a goal and a number of minutes.

How it is served:
- The Functions MCP extension serves every `@app.mcp_tool()` function as a tool at
  /runtime/webhooks/mcp (Streamable HTTP): the tool name comes from the function name,
  the description from its docstring and the input schema from its typed parameters.
- In Azure the endpoint needs the `mcp_extension` system key in the `x-functions-key`
  header, so only callers that have it can use the tools.
- The app scales to zero when idle, so it costs almost nothing.

The plan logic lives in plain functions (`list_goals`, `build_workout`), so it can be
tested without the Functions host:
    python function_app.py
"""

import json
import logging

import azure.functions as func

app = func.FunctionApp()

# Main-block exercises per goal; the plan repeats them in rounds to fill the minutes.
WORKOUTS = {
    "cardio": {
        "description": "Raise the heart rate for endurance.",
        "warm_up": "easy marching or slow cycling, arm circles",
        "main": ["brisk walk or easy jog", "jumping jacks", "step-ups on a low step", "fast walk or cycling"],
        "cool_down": "slow walk, calf and thigh stretches",
    },
    "strength": {
        "description": "Build muscle with bodyweight moves.",
        "warm_up": "arm swings, hip circles, 10 easy squats",
        "main": ["squats", "push-ups (knees or wall if needed)", "glute bridges", "plank"],
        "cool_down": "chest, hip and hamstring stretches",
    },
    "mobility": {
        "description": "Move joints through their range and stretch.",
        "warm_up": "neck rolls, shoulder rolls, gentle torso twists",
        "main": ["cat-cow", "hip openers", "thoracic rotations", "hamstring and calf stretches"],
        "cool_down": "slow breathing, child's pose",
    },
}
MIN_MINUTES = 10
MAX_MINUTES = 90
NOTE = "General fitness ideas, not medical advice. Stop if anything hurts."


def list_goals():
    return {"goals": [{"goal": name, "description": plan["description"]} for name, plan in WORKOUTS.items()]}


def build_workout(goal, minutes):
    """Return a workout plan as a dict, or a dict with an "error" key when the input is
    invalid."""
    goal = str(goal or "").strip().lower()
    if goal not in WORKOUTS:
        return {"error": f"Unknown goal '{goal}'. Use one of: {', '.join(WORKOUTS)}."}
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return {"error": f"minutes must be a whole number from {MIN_MINUTES} to {MAX_MINUTES}."}
    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        return {"error": f"minutes must be from {MIN_MINUTES} to {MAX_MINUTES}, got {minutes}."}

    plan = WORKOUTS[goal]
    warm_up = max(3, round(minutes * 0.15))
    cool_down = max(2, round(minutes * 0.10))
    main = minutes - warm_up - cool_down
    exercises = plan["main"]
    rounds = max(1, main // (2 * len(exercises)))
    seconds_each = main * 60 // (rounds * len(exercises))
    return {
        "goal": goal,
        "total_minutes": minutes,
        "warm_up": {"minutes": warm_up, "do": plan["warm_up"]},
        "main": {
            "minutes": main,
            "rounds": rounds,
            "seconds_per_exercise": seconds_each,
            "exercises": exercises,
        },
        "cool_down": {"minutes": cool_down, "do": plan["cool_down"]},
        "note": NOTE,
    }


@app.mcp_tool()
def list_workout_goals() -> str:
    """List the workout goals that suggest_workout knows, with a short description of each."""  # noqa: W505 - the agent gets this docstring as the tool description
    logging.info("list_workout_goals called")
    return json.dumps(list_goals())


@app.mcp_tool()
@app.mcp_tool_property(arg_name="goal", description=f"Workout goal, one of: {', '.join(WORKOUTS)}.")
@app.mcp_tool_property(
    arg_name="minutes", description=f"Total workout length in minutes, {MIN_MINUTES} to {MAX_MINUTES}."
)
def suggest_workout(goal: str, minutes: int) -> str:
    """Suggest a simple home workout plan (warm-up, main block, cool-down) for a goal and a length in minutes.

    Call list_workout_goals first if you don't know the goals.
    """  # noqa: W505 - the agent gets this docstring as the tool description
    logging.info("suggest_workout called with goal=%r minutes=%r", goal, minutes)
    return json.dumps(build_workout(goal, minutes))


if __name__ == "__main__":
    goals = list_goals()
    print(json.dumps(goals, indent=2))
    for goal, minutes in [("cardio", 30), ("yoga", 30), ("strength", 5)]:
        workout = build_workout(goal, minutes)
        print(json.dumps(workout, indent=2))
