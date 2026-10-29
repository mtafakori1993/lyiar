import json
import logging
import os
import re
import time
import uuid

from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from logging_conf import setup_logging
from lyria_client import LyriaError, generate_lyria
from music_agent import (
    AgentError,
    build_music_prompt,
    call_agent,
)
from outro_agent import (
    OUTRO_FALLBACK_MESSAGE,
    OutroAgentError,
    call_outro_agent,
)
from session_store import SessionStore

setup_logging(config.LOG_LEVEL)
logger = logging.getLogger("lyiar")

app = FastAPI(
    title="Lyiar Music Service",
    description="Music generation service with an LLM orchestration layer (Autonomous Music Creation Agent).",
)

store = SessionStore(config.REDIS_URL, config.SESSION_TTL_SECONDS)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

NEEDS_INPUT_FALLBACK_QUESTION = (
    "کمی بیشتر درباره آهنگی که می‌خوای توضیح بده."
)


class AssistantRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class OutroRequest(BaseModel):
    session_id: Optional[str] = None
    event: str = "playback_finished"


@app.middleware("http")
async def request_context_middleware(request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http_error",
            extra={"request_id": request_id, "path": request.url.path},
        )
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "http_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/generate")
def generate(prompt: str = Form(...), request: Request = None):
    request_id = getattr(request.state, "request_id", None)

    output_path = os.path.join(config.OUTPUT_DIR, "lyria.mp3")

    try:
        lyria_latency_ms = generate_lyria(prompt, output_path)
    except LyriaError as exc:
        logger.warning(
            "lyria_generation_failed",
            extra={
                "request_id": request_id,
                "lyria_model": config.LYRIA_MODEL,
                "generation_status": "error",
                "error": str(exc),
            },
        )
        return {"error": str(exc)}

    logger.info(
        "lyria_generation_success",
        extra={
            "request_id": request_id,
            "lyria_model": config.LYRIA_MODEL,
            "lyria_latency_ms": lyria_latency_ms,
            "generation_status": "generated",
        },
    )

    return {
        "audio": "/audio"
    }


@app.post("/assistant/generate")
def assistant_generate(body: AssistantRequest, request: Request = None):
    request_id = getattr(request.state, "request_id", None)

    message = (body.message or "").strip()
    session_id = body.session_id or uuid.uuid4().hex

    if not SESSION_ID_RE.match(session_id):
        session_id = uuid.uuid4().hex

    session = store.get(session_id) or {
        "session_id": session_id,
        "collected": {},
        "messages": [],
        "explicit_fields": [],
        "last_asked_field": "",
    }

    session.setdefault("collected", {})
    session.setdefault("messages", [])
    session.setdefault("explicit_fields", [])
    session.setdefault("last_asked_field", "")

    logger.info(
        "assistant_request_received",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "message_chars": len(message),
        },
    )

    if not message:
        return {
            "status": "needs_input",
            "message": NEEDS_INPUT_FALLBACK_QUESTION,
            "session_id": session_id,
        }

    try:
        result, llm_latency_ms = call_agent(session, message)
    except AgentError as exc:
        logger.error(
            "music_agent_failed",
            extra={
                "request_id": request_id,
                "session_id": session_id,
                "model": config.MUSIC_AGENT_MODEL,
                "error": str(exc),
            },
        )
        return JSONResponse(
            {
                "status": "error",
                "message": "مشکلی پیش اومد، دوباره امتحان کن.",
                "session_id": session_id,
            },
            status_code=502,
        )

    # The agent is fully autonomous: it never asks for more information and
    # always returns the final structured payload (instruction/style/lyrics
    # with pronunciation). Missing details are filled with creative choices.
    payload = result
    if not isinstance(payload, dict):
        payload = {}

    music_prompt = build_music_prompt(payload)

    logger.info(
        "music_agent_generated",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "model": config.MUSIC_AGENT_MODEL,
            "llm_latency_ms": llm_latency_ms,
            "lyrics_lines": len(payload.get("lyrics") or []),
            "generation_status": "generating",
        },
    )

    session["collected"] = {}
    session["explicit_fields"] = []
    session["last_asked_field"] = ""
    # Persist the full generation context so a later outro request can
    # reference exactly what was created (metadata, prompt, request).
    session["metadata"] = {
        "theme": "",
        "genre": "",
        "mood": "",
        "language": "",
        "vocal_style": "",
    }
    session["music_prompt"] = music_prompt
    session["song_payload"] = payload
    session["music_request"] = message
    session["messages"].append({
        "role": "user",
        "content": message,
    })
    session["messages"].append({
        "role": "assistant",
        "content": "generating",
    })
    store.set(session_id, session)

    output_path = os.path.join(config.OUTPUT_DIR, "{}.mp3".format(session_id))

    logger.info(
        "lyria_generation_started",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "lyria_model": config.LYRIA_MODEL,
            "agent_model": config.MUSIC_AGENT_MODEL,
            "music_prompt_chars": len(music_prompt),
        },
    )

    try:
        lyria_latency_ms = generate_lyria(music_prompt, output_path)
    except LyriaError as exc:
        logger.error(
            "lyria_generation_failed",
            extra={
                "request_id": request_id,
                "session_id": session_id,
                "lyria_model": config.LYRIA_MODEL,
                "generation_status": "error",
                "error": str(exc),
            },
        )
        return JSONResponse(
            {
                "status": "error",
                "message": "مشکلی در ساخت موسیقی پیش اومد.",
                "session_id": session_id,
            },
            status_code=502,
        )

    logger.info(
        "assistant_generating",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "agent_model": config.MUSIC_AGENT_MODEL,
            "lyria_model": config.LYRIA_MODEL,
            "llm_latency_ms": llm_latency_ms,
            "lyria_latency_ms": lyria_latency_ms,
            "generation_status": "generating",
        },
    )

    return {
        "status": "generating",
        "session_id": session_id,
        "prompt": music_prompt,
        "audio": "/audio/{}".format(session_id),
    }


@app.post("/assistant/outro")
def assistant_outro(body: OutroRequest, request: Request = None):
    """Post-playback outro: generate a short spoken message for Galatea
    from the exact song context stored in the session. Only returns the
    text to be voiced — Lyria is never called here."""
    request_id = getattr(request.state, "request_id", None)

    session_id = (body.session_id or "").strip()

    if not session_id or not SESSION_ID_RE.match(session_id):
        return JSONResponse(
            {"status": "error", "message": "invalid session_id"},
            status_code=400,
        )

    session = store.get(session_id)

    if not session:
        logger.warning(
            "outro_session_not_found",
            extra={"request_id": request_id, "session_id": session_id},
        )
        return {"status": "success", "message": OUTRO_FALLBACK_MESSAGE}

    metadata = session.get("metadata") or session.get("collected") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    context = {
        "metadata": metadata,
        "user_request": session.get("music_request") or "",
        "music_prompt": session.get("music_prompt") or "",
        "song_payload": session.get("song_payload") or {},
        "history": session.get("messages") or [],
    }

    logger.info(
        "outro_generation_started",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "genre": metadata.get("genre") or "",
            "mood": metadata.get("mood") or "",
            "theme": metadata.get("theme") or "",
        },
    )

    try:
        text, llm_latency_ms = call_outro_agent(context)
    except OutroAgentError as exc:
        logger.error(
            "outro_agent_failed",
            extra={
                "request_id": request_id,
                "session_id": session_id,
                "model": config.MUSIC_OUTRO_MODEL,
                "error": str(exc),
            },
        )
        text = OUTRO_FALLBACK_MESSAGE
        llm_latency_ms = 0

    logger.info(
        "outro_generation_completed",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "llm_latency_ms": llm_latency_ms,
            "message_length": len(text),
        },
    )

    return {
        "status": "success",
        "message": text,
    }


@app.get("/audio")
def audio():
    return FileResponse(
        os.path.join(config.OUTPUT_DIR, "lyria.mp3"),
        media_type="audio/mpeg"
    )


@app.get("/audio/{session_id}")
def audio_for_session(session_id: str):
    if not SESSION_ID_RE.match(session_id):
        return JSONResponse({"error": "invalid session_id"}, status_code=400)

    path = os.path.join(config.OUTPUT_DIR, "{}.mp3".format(session_id))

    if not os.path.exists(path):
        return JSONResponse({"error": "audio not found"}, status_code=404)

    return FileResponse(path, media_type="audio/mpeg")
