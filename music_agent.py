import json
import logging
import re
import time

import requests

import config

logger = logging.getLogger("lyiar.music_agent")

# ---------------------------------------------------------------------------
# The agent is an autonomous music creation engine: it turns any music
# request directly into the final structured payload and NEVER asks the
# user for more information. Missing details (genre, mood, lyrics, ...) are
# filled with reasonable creative decisions.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are LyiAR Music Creation Engine, not a conversational assistant.
Your only task is to transform music requests into final structured JSON.
Always produce the output.
Never request additional information.
Never ask follow-up questions.
Never respond with conversational guidance.
Never say "tell me the theme", "provide lyrics", or "give me details".

When the request is missing details (only a theme, only a genre, only a mood, a few words, or incomplete lyrics), make reasonable creative decisions yourself and generate everything: the song concept, the style, the complete lyrics, and the pronunciation guides.

Always return ONLY a single valid JSON object:

{
  "instruction": "...",
  "style": "...",
  "lyrics": [
    {"persian": "...", "pronunciation": "..."}
  ]
}

Field rules:

1. "instruction" must always contain the exact sentence:
   "Sing only the Persian lyrics. The pronunciation field is only a phonetic guide. Do not sing Latin pronunciation text. Use it only for accurate Persian pronunciation, accent, articulation, rhythm, and vocal delivery."

2. "style": genre, mood, vocal style, emotion and performance directions. Examples:
   - powerful male vocal
   - emotional gritty voice
   - soft female vocal
   - angry delivery
   - nostalgic feeling
   - cinematic atmosphere
   - energetic chorus

3. "lyrics": an array of objects. Every line MUST contain:
   - "persian": the exact Persian lyric text that should be sung.
   - "pronunciation": a high-quality phonetic guide written in Latin characters.

Pronunciation rules (for the singing model):
- Never use literal translation.
- Never write formal textbook transliteration.
- Write the way a Persian native speaker actually pronounces the lyric.
- Preserve Persian accent, rhythm, syllable length, and vocal flow.
- Use "aa" for long آ, "oo" for long او, "ee" for long ای, "kh" for خ, "gh" for ق/غ, "ch" for چ, "sh" for ش, "zh" for ژ.
- Keep spoken Persian contractions.

Examples:
آروم -> aroom
می‌مونه -> mimoon-e
می‌گیره -> migire
می‌پیچه -> mipiche
هواته -> havata
صداته -> sedata
دلمَه -> delema

- For regional accents (Kermanshahi, Tehrani, Lori, Kurdish, Gilaki, etc.): keep the Persian lyric understandable and adapt the pronunciation only.
- Do not put pronunciation hints inside the Persian lyric. No brackets, parentheses, or Latin words in the lyric.

The lyrics must be complete, original and fit the requested style, mood, theme and language."""

# Sent when the model returns plain text, questions, explanations or invalid
# JSON. The payload is re-generated once with this correction.
CORRECTION_PROMPT = (
    "Return only the required JSON format. Do not explain. Do not ask questions."
)


class AgentError(Exception):
    pass


def _normalize_string(value):
    if value is None:
        return ""
    return str(value).strip()


def validate_payload(payload):
    """Validate the agent's output against the required JSON schema.

    Returns a list of error strings (empty when the payload is valid).
    """
    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]

    errors = []

    if not _normalize_string(payload.get("instruction")):
        errors.append("missing non-empty 'instruction'")

    if not _normalize_string(payload.get("style")):
        errors.append("missing non-empty 'style'")

    lyrics = payload.get("lyrics")
    if not isinstance(lyrics, list) or not lyrics:
        errors.append("missing non-empty 'lyrics' array")
    else:
        for index, line in enumerate(lyrics):
            if not isinstance(line, dict):
                errors.append("lyrics[{}] is not an object".format(index))
                continue
            if not _normalize_string(line.get("persian")):
                errors.append("lyrics[{}] missing 'persian'".format(index))
            if not _normalize_string(line.get("pronunciation")):
                errors.append("lyrics[{}] missing 'pronunciation'".format(index))

    return errors


def build_music_prompt(payload):
    """Render the structured payload into the final Lyria music prompt."""
    payload = payload or {}

    instruction = _normalize_string(payload.get("instruction"))
    style = _normalize_string(payload.get("style"))
    lyrics = payload.get("lyrics")
    if not isinstance(lyrics, list):
        lyrics = []

    parts = []

    if instruction:
        parts.append(instruction)
    if style:
        parts.append("Style: {}".format(style))

    if lyrics:
        parts.append("Lyrics:")
        for index, line in enumerate(lyrics, start=1):
            if not isinstance(line, dict):
                continue
            persian = _normalize_string(line.get("persian"))
            pronunciation = _normalize_string(line.get("pronunciation"))
            if not persian:
                continue
            entry = "{}. {}".format(index, persian)
            if pronunciation:
                entry += " (pronunciation: {})".format(pronunciation)
            parts.append(entry)

    return "\n".join(parts)


def parse_json(content):
    if not isinstance(content, str):
        return None

    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except ValueError:
        return None


def _post_chat(messages, response_format):
    url = "{}/chat/completions".format(config.OPENROUTER_BASE_URL)

    payload = {
        "model": config.MUSIC_AGENT_MODEL,
        "messages": messages,
        "temperature": config.AGENT_TEMPERATURE,
        "max_tokens": config.AGENT_MAX_TOKENS,
    }

    if response_format:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": "Bearer {}".format(config.OPENROUTER_API_KEY),
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )

    return response


def _parse_response_content(response):
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise AgentError("malformed response from model: {}".format(exc))
    return content


def call_agent(session_data, user_message):
    """Transform a music request into the final structured song payload.

    The agent always generates immediately: the conversation history is
    never replayed (a stale "tell me more" assistant message can never
    leak into a future generation). Every request starts fresh with the
    system instruction + the current user request.

    Returns (payload, latency_ms).
    Raises AgentError on transport/upstream/parse/schema failure (after
    one automatic retry with a correction prompt).
    """
    if not config.OPENROUTER_API_KEY:
        raise AgentError("OPENROUTER_API_KEY is not set")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": _normalize_string(user_message),
        },
    ]

    started = time.perf_counter()

    try:
        response = _post_chat(messages, response_format=True)
    except requests.RequestException as exc:
        raise AgentError(str(exc))

    latency_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code != 200:
        body = response.text

        if response.status_code == 400 and "response_format" in body.lower():
            try:
                response = _post_chat(messages, response_format=False)
            except requests.RequestException as exc:
                raise AgentError(str(exc))
            latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            raise AgentError(body)

    parsed = parse_json(_parse_response_content(response))
    errors = validate_payload(parsed) if parsed is not None else ["not valid JSON"]

    if errors:
        # One automatic retry with a correction prompt for plain text,
        # questions, explanations or invalid JSON.
        retry_messages = list(messages)
        retry_messages.append({
            "role": "user",
            "content": CORRECTION_PROMPT,
        })

        try:
            response = _post_chat(retry_messages, response_format=True)
        except requests.RequestException as exc:
            raise AgentError(str(exc))
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            raise AgentError(response.text)

        parsed = parse_json(_parse_response_content(response))
        errors = validate_payload(parsed) if parsed is not None else ["not valid JSON"]

        if errors:
            raise AgentError("invalid payload: {}".format("; ".join(errors)))

    return parsed, latency_ms
