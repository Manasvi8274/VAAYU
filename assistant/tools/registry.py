"""
Tool registry: every action the assistant can take is registered here via @tool.

Safety rule (see plan): any tool touching a sensitive category — finance/payment
apps, credentials, 2FA/OTP, sending messages on the user's behalf, deleting
files, or installing/uninstalling software — MUST be registered with
requires_confirmation=True. The orchestrator checks this flag before dispatch.
"""

from dataclasses import dataclass
from typing import Any, Callable

from guardian import action_rules


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
    # Optional: why this call may not run at all, checked BEFORE asking the user to confirm it (asking
    # "send it?" and then refusing after their yes would be silly) - e.g. WhatsApp's contacts allowlist.
    precheck: Callable[[dict[str, Any]], str | None] | None = None

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
        precheck: Callable[[dict[str, Any]], str | None] | None = None,
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
                precheck=precheck,
            )
            return func

        return decorator

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        return [spec.to_ollama_schema() for spec in self._tools.values()]

    def requires_confirmation(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        """Its own flag, or what it does this time: "click Send" is outward (guardian/action_rules.py)."""
        return self._tools[name].requires_confirmation or action_rules.needs_yes(name, arguments)

    def refusal_before_asking(self, name: str, arguments: dict[str, Any]) -> str | None:
        spec = self._tools.get(name)
        return spec.precheck(arguments) if spec is not None and spec.precheck is not None else None

    def get_confirmation_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        spec = self._tools[name]
        if spec.confirmation_prompt is not None:
            return spec.confirmation_prompt(arguments)
        if arguments.get("app") and arguments.get("target"):  # "click Send" inside an app (windows_apps)
            return f"Should I click {arguments['target']} in {arguments['app']}?"
        args_text = ", ".join(f"{k}: {v}" for k, v in arguments.items())
        return f"Should I go ahead with {spec.name.replace('_', ' ')} ({args_text})?"

    def dispatch(self, name: str, arguments: dict[str, Any], after_content: bool = False,
                 path: str = "") -> dict[str, Any]:
        """after_content: Vaayu read a web page / file / message earlier this turn (outward actions refused)."""
        spec = self._tools.get(name)
        if spec is None:
            return {"ok": False, "error": f"Unknown tool '{name}'"}

        # Local models occasionally hallucinate arguments that aren't in the
        # declared schema — drop anything the tool doesn't actually accept
        # rather than letting an unexpected kwarg crash the call.
        filtered_arguments = repair_arguments(spec, arguments)

        missing = [a for a in spec.parameters.get("required", []) if a not in filtered_arguments]
        if missing:  # real log: "That didn't work - open_app() missing 1 required positional argument: 'app_name'"
            what = missing[0].replace("_name", "").replace("_", " ")
            return {"ok": False, "error": f"I'm not sure which {what} you mean - say it again with the name."}
        refused = action_rules.refusal(name, filtered_arguments, after_content)
        if refused:  # tiers, rate limits, instructions from content (guardian/action_rules.py)
            action_rules.log(name, filtered_arguments, False, refused, path)
            return {"ok": False, "error": refused, "refused": True}
        try:
            result = spec.func(**filtered_arguments)
            action_rules.log(name, filtered_arguments, True, path=path)
            return {"ok": True, "result": result}
        except Exception as exc:  # tool failures shouldn't crash the assistant
            action_rules.log(name, filtered_arguments, False, str(exc), path)
            return {"ok": False, "error": str(exc)}


def repair_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Drops arguments the tool doesn't accept - but first, when exactly one
    required argument is missing and exactly one unknown one was given,
    treats it as a misnamed version of the missing one. Benchmark: the model
    sent open_website(url=...) / read_file(file=...) for url_or_query/path."""
    allowed = spec.parameters.get("properties", {})
    unknown = [k for k in arguments if k not in allowed]
    missing = [k for k in spec.parameters.get("required", []) if k not in arguments]
    fixed = {k: v for k, v in arguments.items() if k in allowed}
    if len(missing) == 1 and len(unknown) == 1:
        fixed[missing[0]] = arguments[unknown[0]]
    return fixed


# Single shared registry instance, imported by orchestrator.py and each tool module.
registry = ToolRegistry()
