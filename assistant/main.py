"""
M1 entry point: text-in/text-out loop wired to the Ollama brain and the tool
registry. Proves the tool-calling pipeline works before voice I/O (M3) is added.
"""

from assistant.log_setup import ensure_output_stream

ensure_output_stream()  # crash-proofs print() against the console's codepage - see log_setup.py

# Importing assistant.tools runs every tool module's @registry.register decorator.
import assistant.tools  # noqa: F401,E402
from assistant.config_schema import load_config  # noqa: E402
from assistant.llm.ollama_client import OllamaBrain  # noqa: E402
from assistant.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from assistant.tools.registry import registry  # noqa: E402


def main() -> None:
    config = load_config()
    brain = OllamaBrain(host=config.llm.host, model=config.llm.model)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Assistant ready (model: {config.llm.model}). Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply, messages = brain.respond(messages, registry)
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
