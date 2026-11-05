"""Voice tools for background jobs (assistant/agent.py)."""

from assistant import agent
from assistant.tools.registry import registry


@registry.register(
    name="do_task",
    description="Start a job that needs several steps (research something and put it in a document, compare "
                "options, gather information) - runs in the background, Vaayu reports when it's done.",
    parameters={"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
)
def do_task(task: str) -> str:
    return agent.start(task)


@registry.register(name="task_status", description="How the background job is going.",
                   parameters={"type": "object", "properties": {}})
def task_status() -> str:
    return agent.status()
