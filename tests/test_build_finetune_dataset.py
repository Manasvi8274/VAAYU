"""
scripts/build_finetune_dataset.py - the fine-tuning dataset generator.
Mainly tests that _tool_call's validation against the live registry
actually catches bad entries (so a future edit to this script that
introduces a typo'd argument name fails loudly, not silently), and that
the specific bug-case examples (the whole point of this dataset) are
actually present and correctly labeled.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_finetune_dataset as bfd  # noqa: E402


def test_tool_call_accepts_valid_arguments():
    result = bfd._tool_call("open_app", {"app_name": "notepad"})
    assert result == {"function": {"name": "open_app", "arguments": {"app_name": "notepad"}}}


def test_tool_call_rejects_unknown_tool():
    with pytest.raises(ValueError, match="unknown tool"):
        bfd._tool_call("not_a_real_tool", {})


def test_tool_call_rejects_unexpected_argument():
    with pytest.raises(ValueError, match="unexpected args"):
        bfd._tool_call("open_app", {"app_name": "notepad", "made_up_arg": "x"})


def test_tool_call_rejects_missing_required_argument():
    with pytest.raises(ValueError, match="missing required args"):
        bfd._tool_call("open_app", {})


def test_build_examples_produces_only_valid_messages_shapes():
    examples = bfd.build_examples()
    assert len(examples) > 50
    for ex in examples:
        messages = ex["messages"]
        assert messages[0]["role"] == "system" and messages[0]["content"]
        assert messages[1]["role"] == "user" and messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        # either a real tool call or a plain conversational reply, never both/neither
        has_tool_calls = bool(messages[2].get("tool_calls"))
        has_content = bool(messages[2].get("content"))
        assert has_tool_calls or has_content


def _find(examples, user_text):
    for ex in examples:
        if ex["messages"][1]["content"] == user_text:
            return ex["messages"][2]
    raise AssertionError(f"No example found for: {user_text!r}")


@pytest.mark.parametrize("user_text,expected_tool,expected_args", [
    # each of these is a real bug found and fixed this session - the whole
    # reason this dataset exists is to reinforce not repeating them.
    ("close the third tab", "close_browser_tab", {"position": 3}),
    ("close the brave browser", "close_app", {"app_name": "brave"}),
    ("play fairytale music", "play_video_on_youtube", {"query": "fairytale music"}),
    ("increase the youtube volume", "set_video_volume", {"direction": "up"}),
    ("turn up the system volume", "control_media", {"action": "volume_up"}),
    (
        "type 9876543210",
        "type_sensitive_field",
        {"value": "9876543210", "field_description": "phone number field"},
    ),
    ("type hello world", "type_text", {"text": "hello world"}),
])
def test_key_bug_case_examples_are_present_and_correct(user_text, expected_tool, expected_args):
    examples = bfd.build_examples()
    assistant_msg = _find(examples, user_text)
    call = assistant_msg["tool_calls"][0]["function"]
    assert call["name"] == expected_tool
    assert call["arguments"] == expected_args


def test_non_tool_examples_have_no_tool_calls():
    examples = bfd.build_examples()
    assistant_msg = _find(examples, "how are you")
    assert not assistant_msg.get("tool_calls")
    assert assistant_msg["content"]


def test_main_writes_valid_jsonl(tmp_path, monkeypatch):
    import json

    out_path = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(bfd, "_OUT_PATH", out_path)

    bfd.main()

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) > 50
    for line in lines:
        parsed = json.loads(line)  # every line must be valid, standalone JSON
        assert "messages" in parsed
