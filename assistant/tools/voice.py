"""'Vaayu, change your voice' / 'use the George voice' - switches the Kokoro voice live and remembers it."""

from assistant.tools.registry import registry


@registry.register(
    name="set_voice",
    description="Change Vaayu's speaking voice. voice: a name (echo, lewis, george, daniel, fable, adam, omega, psi, "
                "michael, puck, heart, emma) or 'next' to try the next one.",
    parameters={"type": "object", "properties": {"voice": {"type": "string"}}, "required": ["voice"]},
)
def set_voice(voice: str) -> str:
    from assistant.tts import kokoro_engine
    engine = kokoro_engine.ACTIVE
    if engine is None:
        raise RuntimeError("I'm using the Windows voice right now - the voice choice is for the natural voice.")
    names = list(kokoro_engine.VOICES)
    want = voice.strip().lower()
    if want in ("next", "another", "different", ""):
        current = next((n for n, v in kokoro_engine.VOICES.items() if v == engine.voice), None)
        want = names[(names.index(current) + 1) % len(names)] if current else names[0]
    if want not in kokoro_engine.VOICES:
        raise ValueError(f"I don't have a voice called {voice}. Try: {', '.join(names[:6])}.")
    engine.set_voice(kokoro_engine.VOICES[want])
    return f"Okay, this is the {want.title()} voice. Say 'change your voice' to hear the next one."
