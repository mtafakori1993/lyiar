"""Deterministic tests for the autonomous music creation agent.

The agent is a music creation engine, NOT a conversational assistant: it
always turns the request directly into the final structured JSON payload
(instruction / style / lyrics with pronunciation), never asks follow-up
questions, and never enters a clarification loop. These tests pin that
behaviour end to end.
"""

import json

import pytest
from fastapi.testclient import TestClient

import config
import main
from music_agent import (
    AgentError,
    CORRECTION_PROMPT,
    SYSTEM_PROMPT,
    build_music_prompt,
    call_agent,
    parse_json,
    validate_payload,
)


def _valid_payload():
    return {
        "instruction": (
            "Sing only the Persian lyrics. The pronunciation field is only "
            "a phonetic guide. Do not sing Latin pronunciation text."
        ),
        "style": "Epic rock with a powerful male vocal, cinematic atmosphere, energetic chorus.",
        "lyrics": [
            {"persian": "از دل شب برمی‌گردم به اوج", "pronunciation": "az dele shab barmigardam be ooj"},
            {"persian": "دستم رو بگیر، بالا بکش", "pronunciation": "dastamo begir, baalaa bekesh"},
        ],
    }


def _incomplete_payload():
    return {
        "instruction": "",
        "style": "Pop",
        "lyrics": [],
    }


# ---------------------------------------------------------------------------
# Unit tests: schema validation
# ---------------------------------------------------------------------------

def test_validate_payload_accepts_valid_payload():
    assert validate_payload(_valid_payload()) == []


def test_validate_payload_rejects_missing_instruction():
    payload = _valid_payload()
    payload["instruction"] = ""
    errors = validate_payload(payload)
    assert any("instruction" in e for e in errors)


def test_validate_payload_rejects_missing_lyrics():
    payload = _valid_payload()
    payload["lyrics"] = []
    errors = validate_payload(payload)
    assert any("lyrics" in e for e in errors)


def test_validate_payload_rejects_line_without_pronunciation():
    payload = _valid_payload()
    payload["lyrics"].append({"persian": "بدون تلفظ"})
    errors = validate_payload(payload)
    assert any("pronunciation" in e for e in errors)


def test_validate_payload_rejects_non_object():
    assert validate_payload("hello") != []
    assert validate_payload(None) != []


def test_build_music_prompt_contains_instruction_style_and_lyrics():
    prompt = build_music_prompt(_valid_payload())
    assert "Sing only the Persian lyrics" in prompt
    assert "Style:" in prompt
    assert "1. از دل شب برمی‌گردم به اوج" in prompt
    assert "(pronunciation: az dele shab barmigardam be ooj)" in prompt


def test_build_music_prompt_handles_empty_payload():
    assert build_music_prompt({}) == ""


def test_parse_json_handles_fenced_blocks():
    content = "```json\n{}\n```".format(json.dumps(_valid_payload(), ensure_ascii=False))
    assert parse_json(content) == _valid_payload()


def test_system_prompt_forbids_asking():
    assert "Never request additional information" in SYSTEM_PROMPT
    assert "Never ask follow-up questions" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# call_agent: retry once with a correction prompt on invalid output
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.text = content
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture()
def api_key():
    old = config.OPENROUTER_API_KEY
    config.OPENROUTER_API_KEY = "test-key"
    yield
    config.OPENROUTER_API_KEY = old


def test_call_agent_returns_valid_payload(monkeypatch, api_key):
    calls = []

    def fake_post(messages, response_format):
        calls.append(messages)
        return _FakeResponse(200, json.dumps(_valid_payload(), ensure_ascii=False))

    monkeypatch.setattr("music_agent._post_chat", fake_post)

    payload, latency = call_agent({}, "یک آهنگ راک حماسی درباره بازگشت به اوج بساز")

    assert payload == _valid_payload()
    assert latency >= 0
    assert len(calls) == 1
    assert calls[0][0]["role"] == "system"
    assert "Never request additional information" in calls[0][0]["content"]


def test_call_agent_retries_once_on_invalid_json(monkeypatch, api_key):
    calls = []

    def fake_post(messages, response_format):
        calls.append(messages)
        if len(calls) == 1:
            return _FakeResponse(200, "Can you tell me the theme please?")
        return _FakeResponse(200, json.dumps(_valid_payload(), ensure_ascii=False))

    monkeypatch.setattr("music_agent._post_chat", fake_post)

    payload, _ = call_agent({}, "یه آهنگ شاد کردی بساز")

    assert payload == _valid_payload()
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "user"
    assert calls[1][-1]["content"] == CORRECTION_PROMPT


def test_call_agent_fails_after_two_invalid_responses(monkeypatch, api_key):
    def fake_post(messages, response_format):
        return _FakeResponse(200, "Not JSON at all")

    monkeypatch.setattr("music_agent._post_chat", fake_post)

    with pytest.raises(AgentError):
        call_agent({}, "یه آهنگ بساز")


def test_call_agent_incomplete_schema_retries(monkeypatch, api_key):
    calls = []

    def fake_post(messages, response_format):
        calls.append(messages)
        if len(calls) == 1:
            return _FakeResponse(200, json.dumps(_incomplete_payload(), ensure_ascii=False))
        return _FakeResponse(200, json.dumps(_valid_payload(), ensure_ascii=False))

    monkeypatch.setattr("music_agent._post_chat", fake_post)

    payload, _ = call_agent({}, "یه آهنگ غمگین درباره دلتنگی")
    assert payload == _valid_payload()
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# End-to-end session flow (POST /assistant/generate)
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_client():
    config.REDIS_URL = "redis://127.0.0.1:1/0"
    main.store = main.SessionStore(config.REDIS_URL, 300)
    client = TestClient(main.app)
    yield client
    main.store._memory.clear()


def _post_generate(client, session_id, message):
    return client.post(
        "/assistant/generate",
        json={"session_id": session_id, "message": message},
    )


def test_generate_epic_rock_song_immediately(agent_client, monkeypatch):
    monkeypatch.setattr(main, "call_agent", lambda session, msg: (_valid_payload(), 40))
    monkeypatch.setattr(main, "generate_lyria", lambda prompt, path: 100)

    resp = _post_generate(
        agent_client, "sess-rock", "یک آهنگ راک حماسی درباره بازگشت به اوج بساز"
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "generating"
    assert "needs_input" not in body
    assert body["prompt"]
    assert body["audio"] == "/audio/sess-rock"

    sess = main.store.get("sess-rock")
    payload = sess["song_payload"]
    assert payload["instruction"]
    assert isinstance(payload["lyrics"], list) and len(payload["lyrics"]) > 0
    assert all(line.get("pronunciation") for line in payload["lyrics"])


def test_generate_happy_kurdish_song_no_clarification(agent_client, monkeypatch):
    kurdish_payload = {
        "instruction": _valid_payload()["instruction"],
        "style": "Happy Kurdish folk with a warm male vocal, energetic chorus.",
        "lyrics": [
            {"persian": "دلم جاره، دلم خوشه امروز", "pronunciation": "delem jare, delem khushe emrooz"},
            {"persian": "برقص و بخند، دنیا مال ماست", "pronunciation": "beraghs o bekhand, donyaa maale maast"},
        ],
    }
    monkeypatch.setattr(main, "call_agent", lambda session, msg: (kurdish_payload, 30))
    monkeypatch.setattr(main, "generate_lyria", lambda prompt, path: 90)

    resp = _post_generate(agent_client, "sess-kurdish", "یه آهنگ شاد کردی بساز")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "generating"
    assert "needs_input" not in body

    sess = main.store.get("sess-kurdish")
    assert len(sess["song_payload"]["lyrics"]) == 2
    assert sess["song_payload"]["lyrics"][0]["pronunciation"]


def test_generate_vague_request_still_generates(agent_client, monkeypatch):
    # A bare request with zero details must still produce a song — the
    # agent fills in the creative decisions itself.
    monkeypatch.setattr(main, "call_agent", lambda session, msg: (_valid_payload(), 25))
    monkeypatch.setattr(main, "generate_lyria", lambda prompt, path: 80)

    resp = _post_generate(agent_client, "sess-vague", "یه آهنگ بساز")

    assert resp.status_code == 200
    assert resp.json()["status"] == "generating"
    assert "needs_input" not in resp.json()


def test_generate_agent_error_returns_502(agent_client, monkeypatch):
    def failing_agent(session, msg):
        raise AgentError("upstream down")

    monkeypatch.setattr(main, "call_agent", failing_agent)

    resp = _post_generate(agent_client, "sess-fail", "یه آهنگ بساز")

    assert resp.status_code == 502
    assert resp.json()["status"] == "error"


def test_generate_lyria_error_returns_502(agent_client, monkeypatch):
    from lyria_client import LyriaError

    monkeypatch.setattr(main, "call_agent", lambda session, msg: (_valid_payload(), 40))
    monkeypatch.setattr(
        main, "generate_lyria",
        lambda prompt, path: (_ for _ in ()).throw(LyriaError("no audio")),
    )

    resp = _post_generate(agent_client, "sess-lyria-fail", "یه آهنگ بساز")

    assert resp.status_code == 502
    assert resp.json()["status"] == "error"
