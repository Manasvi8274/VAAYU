from assistant.storage.db import get_connection
from assistant.tools.registry import registry


@registry.register(
    name="add_note",
    description=(
        "Save ONE new short note for later, e.g. 'note that I need to call the dentist', 'remember "
        "to buy groceries'. This WRITES a new note - never use it for a request to see/read/hear "
        "existing notes ('what are my notes', 'what are my recent notes', 'list my notes') - use "
        "list_notes for those instead. Never combine multiple prior notes' text into one call here -"
        "each add_note call saves exactly the one new thing the user just said, nothing else."
    ),
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The note text to save."}},
        "required": ["text"],
    },
)
def add_note(text: str) -> str:
    conn = get_connection()
    conn.execute("INSERT INTO notes (text) VALUES (?)", (text,))
    conn.commit()
    conn.close()
    return "Note saved."


@registry.register(
    name="list_notes",
    description=(
        "List the most recently saved notes - use this for ANY request to see/read/hear existing "
        "notes: 'what are my notes', 'what are my recent notes', 'list my notes', 'read back my "
        "notes'. This is a READ - it never saves/modifies anything. Do NOT use add_note for these "
        "phrasings, even though both tools are about notes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum number of notes to return (default 5)."}
        },
        "required": [],
    },
)
def list_notes(limit: int = 5) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT text, created_at FROM notes ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"text": row[0], "created_at": row[1]} for row in rows]
