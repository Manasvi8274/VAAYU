import pytest

from assistant.tools.registry import ToolRegistry

_EMPTY_PARAMS = {"type": "object", "properties": {}, "required": []}


def test_register_and_schema():
    reg = ToolRegistry()

    @reg.register(
        name="echo",
        description="Echo back the given text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    def echo(text: str) -> str:
        return text

    schemas = reg.get_ollama_tools()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"
    assert schemas[0]["function"]["parameters"]["required"] == ["text"]


def test_dispatch_success():
    reg = ToolRegistry()

    @reg.register(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    def add(a: float, b: float) -> float:
        return a + b

    assert reg.dispatch("add", {"a": 2, "b": 3}) == {"ok": True, "result": 5}


def test_dispatch_unknown_tool():
    reg = ToolRegistry()
    result = reg.dispatch("does_not_exist", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


def test_dispatch_filters_hallucinated_arguments():
    """A local model sometimes invents arguments that aren't in the schema;
    dispatch must drop them instead of crashing on an unexpected kwarg."""
    reg = ToolRegistry()

    @reg.register(name="ping", description="Ping.", parameters=_EMPTY_PARAMS)
    def ping() -> str:
        return "pong"

    assert reg.dispatch("ping", {"unexpected_arg": "value"}) == {"ok": True, "result": "pong"}


def test_dispatch_catches_tool_exceptions():
    reg = ToolRegistry()

    @reg.register(name="boom", description="Always fails.", parameters=_EMPTY_PARAMS)
    def boom() -> str:
        raise ValueError("kaboom")

    result = reg.dispatch("boom", {})
    assert result["ok"] is False
    assert "kaboom" in result["error"]


def test_requires_confirmation_flag_defaults_false():
    reg = ToolRegistry()

    @reg.register(name="safe_tool", description="Safe.", parameters=_EMPTY_PARAMS)
    def safe_tool() -> str:
        return "ok"

    assert reg.requires_confirmation("safe_tool") is False


def test_requires_confirmation_flag_can_be_set():
    reg = ToolRegistry()

    @reg.register(
        name="dangerous_tool",
        description="Dangerous.",
        parameters=_EMPTY_PARAMS,
        requires_confirmation=True,
    )
    def dangerous_tool() -> str:
        return "did it"

    assert reg.requires_confirmation("dangerous_tool") is True


def test_duplicate_registration_raises():
    reg = ToolRegistry()

    @reg.register(name="dup", description="First.", parameters=_EMPTY_PARAMS)
    def first() -> str:
        return "first"

    with pytest.raises(ValueError):

        @reg.register(name="dup", description="Second.", parameters=_EMPTY_PARAMS)
        def second() -> str:
            return "second"
