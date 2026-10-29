import json
import logging
import time

import requests

import config

logger = logging.getLogger("lyiar.outro_agent")

# Spoken after a generated song finishes playing. Used when the session or
# its metadata is missing, and as the safe fallback when the LLM fails.
OUTRO_FALLBACK_MESSAGE = (
    "امیدوارم از این آهنگ لذت برده باشی. دوست داری یه آهنگ دیگه هم بسازیم؟"
)

OUTRO_SYSTEM_PROMPT = """You are Galatea, a warm Persian voice assistant.

Your job is to speak after a generated song finishes playing.

You already know the song that was created from the provided context.
Never pretend you don't know it.
Never explain technical details about AI or Lyria.

Your response must:
- be short (5-12 seconds when spoken)
- sound natural in Tehran Persian
- feel emotional and friendly
- mention something specific about the created song when possible
- sometimes ask a natural follow-up question, but do not always ask one
- never repeat the whole song description
- never say "I generated this song"

Examples:

Context:
theme: birthday for daughter
genre: Persian pop
mood: happy

Good:
"چه آهنگ شادی شد! برای تولد هایدی یه حس قشنگ و پرانرژی داره. امیدوارم کلی خاطره خوب بسازید."

Context:
romantic song

Good:
"این یکی یه حس آروم و عاشقانه داشت. دوست داری یه نسخه متفاوت‌تر هم امتحان کنیم؟"

Return only the spoken Persian text.
"""


class OutroAgentError(Exception):
    pass


def _normalize_string(value):
    if value is None:
        return ""
    return str(value).strip()


def build_context_message(context):
    """Render the generation context into the user-facing prompt text."""
    if not isinstance(context, dict):
        context = {}

    metadata = context.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    history = context.get("history") or []
    song_payload = context.get("song_payload") or {}
    if not isinstance(song_payload, dict):
        song_payload = {}

    return "\n".join([
        "Context:",
        "",
        "Song metadata:",
        json.dumps(metadata, ensure_ascii=False),
        "",
        "Song payload:",
        json.dumps(song_payload, ensure_ascii=False),
        "",
        "Original request:",
        _normalize_string(context.get("user_request")),
        "",
        "Generated Lyria prompt:",
        _normalize_string(context.get("music_prompt")),
        "",
        "Conversation history:",
        json.dumps(history, ensure_ascii=False),
        "",
        "Generate the outro message now.",
    ])


def _post_chat(messages, response_format=True):
    url = "{}/chat/completions".format(config.OPENROUTER_BASE_URL)

    payload = {
        "model": config.MUSIC_OUTRO_MODEL,
        "messages": messages,
        "temperature": config.MUSIC_OUTRO_TEMPERATURE,
        "max_tokens": config.MUSIC_OUTRO_MAX_TOKENS,
    }

    if response_format:
        payload["response_format"] = {"type": "text"}

    headers = {
        "Authorization": "Bearer {}".format(config.OPENROUTER_API_KEY),
        "Content-Type": "application/json",
    }

    return requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=config.MUSIC_OUTRO_TIMEOUT_SECONDS,
    )


def call_outro_agent(context):
    """Generate the short spoken outro message from the generation context.

    Returns (text, latency_ms). Raises OutroAgentError on any transport,
    upstream or empty-response failure.
    """
    if not config.OPENROUTER_API_KEY:
        raise OutroAgentError("OPENROUTER_API_KEY is not set")

    messages = [
        {
            "role": "system",
            "content": OUTRO_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_context_message(context),
        },
    ]

    started = time.perf_counter()

    try:
        response = _post_chat(messages)
    except requests.RequestException as exc:
        raise OutroAgentError(str(exc))

    latency_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code != 200:
        body = response.text

        if response.status_code == 400 and "response_format" in body.lower():
            try:
                response = _post_chat(messages, response_format=False)
            except requests.RequestException as exc:
                raise OutroAgentError(str(exc))
            latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            raise OutroAgentError(body)

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise OutroAgentError("malformed response from model: {}".format(exc))

    text = _normalize_string(content)

    if not text:
        raise OutroAgentError("model returned empty response")

    return text, latency_ms
