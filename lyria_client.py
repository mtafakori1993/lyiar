import base64
import json
import logging
import os
import time

import requests

import config

logger = logging.getLogger("lyiar.lyria")


class LyriaError(Exception):
    pass


def generate_lyria(music_prompt, output_path):
    """Generate audio from a music prompt and write it to output_path.

    Returns the elapsed time in milliseconds.
    Raises LyriaError on any upstream or decoding failure.
    """
    if not config.OPENROUTER_API_KEY:
        raise LyriaError("OPENROUTER_API_KEY is not set")

    url = "{}/chat/completions".format(config.OPENROUTER_BASE_URL)

    payload = {
        "model": config.LYRIA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": music_prompt,
            }
        ],
        "modalities": [
            "audio"
        ],
        "audio": {
            "format": "mp3"
        },
        "stream": True,
    }

    headers = {
        "Authorization": "Bearer {}".format(config.OPENROUTER_API_KEY),
        "Content-Type": "application/json",
    }

    started = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=config.HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LyriaError(str(exc))

    latency_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = response.text
        raise LyriaError(error)

    audio_chunks = []

    for line in response.iter_lines():

        if not line:
            continue

        text = line.decode("utf-8", errors="ignore")

        if not text.startswith("data: "):
            continue

        data = text[len("data: "):].strip()

        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except ValueError:
            continue

        choices = chunk.get("choices") or []

        if not choices:
            continue

        delta = choices[0].get("delta") or {}
        audio = delta.get("audio") or {}

        if audio.get("data"):
            audio_chunks.append(audio["data"])

        message = choices[0].get("message") or {}
        content = message.get("content")

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "audio":
                    part_audio = part.get("audio") or {}
                    part_data = part_audio.get("data") or part.get("data")
                    if part_data:
                        audio_chunks.append(part_data)

    if not audio_chunks:
        raise LyriaError("audio not found")

    audio_b64 = "".join(audio_chunks)
    audio = base64.b64decode(audio_b64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(audio)

    logger.info(
        "lyria_generated",
        extra={
            "lyria_latency_ms": latency_ms,
            "lyria_model": config.LYRIA_MODEL,
        },
    )

    return latency_ms
