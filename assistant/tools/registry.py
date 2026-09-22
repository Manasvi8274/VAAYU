"""
Tool registry: every action the assistant can take is registered here via @tool.

Safety rule (see plan): any tool touching a sensitive category — finance/payment
apps, credentials, 2FA/OTP, sending messages on the user's behalf, deleting
files, or installing/uninstalling software — MUST be registered with
requires_confirmation=True. The orchestrator checks this flag before dispatch.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    requires_confirmation: bool = False
    # Optional: build a human-readable confirmation question from the call's
    # arguments (e.g. quoting back the actual message text for a WhatsApp
    # send) - falls back to a generic prompt naming the tool/arguments if
    # not given.
    confirmation_prompt: Callable[[dict[str, Any]], str] | None = None

    def to_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        requires_confirmation: bool = False,
        confirmation_prompt: Callable[[dict[str, Any]], str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered")
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                requires_confirmation=requires_confirmation,
                confirmation_prompt=confirmation_prompt,
            )
            return func

        return decorator

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        return [spec.to_ollama_schema() for spec in self._tools.values()]

    def requires_confirmation(self, name: str) -> bool:
        return self._tools[name].requires_confirmation

    def get_confirmation_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        spec = self._tools[name]
        if spec.confirmation_prompt is not None:
            return spec.confirmation_prompt(arguments)
        args_text = ", ".join(f"{k}: {v}" for k, v in arguments.items())
        return f"Should I go ahead with {spec.name.replace('_', ' ')} ({args_text})?"

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            return {"ok": False, "error": f"Unknown tool '{name}'"}

        # Local models occasionally hallucinate arguments that aren't in the
        # declared schema — drop anything the tool doesn't actually accept
        # rather than letting an unexpected kwarg crash the call.
        allowed = spec.parameters.get("properties", {}).keys()
        filtered_arguments = {k: v for k, v in arguments.items() if k in allowed}

        try:
            result = spec.func(**filtered_arguments)
            return {"ok": True, "result": result}
        except Exception as exc:  # tool failures shouldn't crash the assistant
            return {"ok": False, "error": str(exc)}


# Single shared registry instance, imported by orchestrator.py and each tool module.
registry = ToolRegistry()
