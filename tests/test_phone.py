"""Vaayu from the phone (assistant/phone.py) - against a fake Telegram, never the real one."""
import json
import threading
import time

import pytest

from assistant import phone


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def __call__(self, method, files=None, **params):
        self.sent.append((method, params))
        return {"ok": True, "result": []}

    def texts(self):
        return [p["text"] for m, p in self.sent if m == "sendMessage"]


def _msg(chat, text):
    return {"update_id": 1, "message": {"chat": {"id": chat}, "text": text}}


@pytest.fixture
def bridge(tmp_path):
    path = tmp_path / "telegram.json"
    path.write_text(json.dumps({"token": "T"}))
    handled = []
    tg = FakeTelegram()
    b = phone.TelegramBridge(lambda text, confirm, audio: handled.append(text) or f"Did: {text}",
                             config_path=path, call=tg, log=lambda *_: None)
    b.handled, b.tg = handled, tg
    return b


def _wait(cond):
    deadline = time.time() + 2
    while not cond() and time.time() < deadline:
        time.sleep(0.01)
    return cond()


def test_only_the_paired_chat_is_obeyed(bridge):
    code = bridge.cfg["pair_code"]
    bridge.on_update(_msg(111, "open chrome"))            # a stranger before pairing: nothing
    bridge.on_update(_msg(111, "/start 000000"))          # wrong code: nothing
    assert bridge.tg.sent == [] and bridge.handled == []
    bridge.on_update(_msg(222, f"/start {code}"))
    assert "Paired" in bridge.tg.texts()[0]
    assert json.loads(bridge.path.read_text())["chat_id"] == 222
    bridge.on_update(_msg(111, "open chrome"))            # another chat: still nothing
    bridge.on_update(_msg(222, "open chrome"))
    assert _wait(lambda: "Did: open chrome" in bridge.tg.texts())
    assert bridge.handled == ["open chrome"]


def test_an_outward_action_is_asked_in_the_chat(bridge, monkeypatch):
    bridge.cfg["chat_id"] = 222
    from assistant.tools.registry import registry
    monkeypatch.setattr(registry, "get_confirmation_prompt", lambda n, a: "Send 'hi' to Saved Messages?")
    result = []
    t = threading.Thread(target=lambda: result.append(bridge.confirm("send_whatsapp_message", {})))
    t.start()
    assert _wait(lambda: any("Send 'hi'" in s for s in bridge.tg.texts()))
    bridge.on_update(_msg(222, "yes"))
    t.join(2)
    assert result == [True] and bridge.handled == []      # the "yes" answered the question, it wasn't a command


def test_no_token_no_bridge(tmp_path):
    assert phone.TelegramBridge(lambda *a: "", config_path=tmp_path / "none.json").start() is None


def test_a_line_typed_in_the_orb_panel_is_shown_done_and_spoken(monkeypatch):
    import queue
    from assistant import orchestrator
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.gui_queue, said = queue.Queue(), []
    o.handle_remote = lambda text, confirm, audio=None: f"Did: {text}"
    o._speak = said.append
    o._voice_confirm = lambda *a: False
    o.handle_typed("open chrome")
    shown = [m["who"] + ":" + m["text"] for m in iter(lambda: o.gui_queue.get() if not o.gui_queue.empty() else None, None)]
    assert shown == ["you:open chrome", "vaayu:Did: open chrome"] and said == ["Did: open chrome"]
