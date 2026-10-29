import json

import pytest
from fastapi.testclient import TestClient

import main
import config
from outro_agent import OUTRO_FALLBACK_MESSAGE, OutroAgentError

# An unreachable Redis URL forces SessionStore's in-memory backend, so the
# tests never depend on a live Redis (deterministic, self-contained).
config.REDIS_URL = "redis://127.0.0.1:1/0"

# Recreate the module-level store with the unreachable URL so every test
# exercises a clean in-memory backend.
main.store = main.SessionStore(config.REDIS_URL, 300)

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _clean_sessions():
    yield
    main.store._memory.clear()


def _seed_session(session_id, **overrides):
    session = {
        "session_id": session_id,
        "collected": {
            "theme": "تولد دختر",
            "genre": "پاپ فارسی",
            "mood": "شاد",
            "language": "Persian",
        },
        "metadata": {
            "theme": "تولد دختر",
            "genre": "پاپ فارسی",
            "mood": "شاد",
            "language": "Persian",
        },
        "music_prompt": "A joyful Persian pop birthday song",
        "music_request": "یه آهنگ شاد برای تولد دخترم",
        "messages": [
            {"role": "user", "content": "یه آهنگ شاد برای تولد دخترم"},
            {"role": "assistant", "content": "generating"},
        ],
    }
    session.update(overrides)
    main.store.set(session_id, session)


def _post_outro(session_id):
    return client.post(
        "/assistant/outro",
        json={"session_id": session_id, "event": "music_finished"},
    )


def test_outro_returns_valid_response_for_song_context(monkeypatch):
    _seed_session("sess-birthday-1")
    monkeypatch.setattr(
        main,
        "call_outro_agent",
        lambda context: ("چه آهنگ شادی شد! امیدوارم تولد دخترت پر از خاطره باشه.", 120),
    )

    resp = _post_outro("sess-birthday-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["message"] == "چه آهنگ شادی شد! امیدوارم تولد دخترت پر از خاطره باشه."


def test_outro_passes_song_context_to_agent(monkeypatch):
    _seed_session("sess-birthday-2")
    captured = {}

    def fake_call(context):
        captured.update(context)
        return "یه آهنگ پرانرژی شد!", 100

    monkeypatch.setattr(main, "call_outro_agent", fake_call)

    _post_outro("sess-birthday-2")

    assert captured["user_request"] == "یه آهنگ شاد برای تولد دخترم"
    assert captured["music_prompt"] == "A joyful Persian pop birthday song"
    assert captured["metadata"]["theme"] == "تولد دختر"
    assert len(captured["history"]) == 2


def test_outro_invalid_session_id_rejected():
    resp = _post_outro("bad session id!!")

    assert resp.status_code == 400
    assert resp.json()["status"] == "error"


def test_outro_unknown_session_returns_fallback(monkeypatch):
    def _should_not_be_called(_):
        raise AssertionError("the LLM must not be called for an unknown session")

    monkeypatch.setattr(main, "call_outro_agent", _should_not_be_called)

    resp = _post_outro("sess-never-created")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["message"] == OUTRO_FALLBACK_MESSAGE


def test_outro_agent_failure_returns_fallback(monkeypatch):
    _seed_session("sess-birthday-3")
    monkeypatch.setattr(
        main,
        "call_outro_agent",
        lambda context: (_ for _ in ()).throw(OutroAgentError("upstream down")),
    )

    resp = _post_outro("sess-birthday-3")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["message"] == OUTRO_FALLBACK_MESSAGE


def test_outro_missing_session_id_rejected():
    resp = client.post(
        "/assistant/outro", json={"event": "music_finished"}
    )

    assert resp.status_code == 400
    assert resp.json()["status"] == "error"


def test_outro_synthetic_empty_result_falls_back(monkeypatch):
    """The endpoint must never return an empty message: an empty/blank
    generation result degrades to the fallback just like a failure."""
    _seed_session("sess-birthday-4")
    monkeypatch.setattr(
        main,
        "call_outro_agent",
        lambda context: (_ for _ in ()).throw(OutroAgentError("model returned empty response")),
    )

    resp = _post_outro("sess-birthday-4")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["message"] == OUTRO_FALLBACK_MESSAGE
