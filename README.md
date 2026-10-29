# Lyiar Music Service

A standalone FastAPI microservice that generates music from a text prompt using Google's Lyria 3 Clip model via [OpenRouter](https://openrouter.ai), with an LLM orchestration layer (the **Music Creation Agent**) that autonomously turns any music request directly into a complete song payload (instruction, style, lyrics with pronunciation guides) and generates it.

The service is fully independent and has no coupling to any other project.

## Architecture

```
User Request
    |
    v
POST /assistant/generate
    |
    v
Music Creation Agent (LLM via OpenRouter, google/gemini-3.1-flash-lite)
    |  transforms the request directly into structured JSON:
    |  {instruction, style, lyrics[{persian, pronunciation}]}
    |  (never asks follow-up questions — fills missing details itself)
    v
Generate Lyria Prompt
    |
    v
Lyria 3 Clip (google/lyria-3-clip-preview)
    |
    v
Generated Audio
```

- **`config.py`** — All settings are read from environment variables.
- **`music_agent.py`** — The autonomous music creation agent. Turns any music request directly into the final structured JSON payload (instruction, style, lyrics with Persian pronunciation guides). Never asks follow-up questions; fills missing details with creative decisions. Validates the output schema and retries once with a correction prompt if the model returns plain text or invalid JSON.
- **`outro_agent.py`** — Post-playback outro generator. Given the stored generation context (metadata, song payload, original request, Lyria prompt, conversation history), it produces a short spoken Persian message for Galatea to voice.
- **`session_store.py`** — Session memory. Redis-backed by default with a 30-minute TTL; falls back to in-memory storage if Redis is unavailable.
- **`lyria_client.py`** — The Lyria 3 Clip generation pipeline (streaming base64 audio decode).
- **`logging_conf.py`** — Structured JSON logging.
- **`main.py`** — FastAPI routes and request/response wiring.
- **`static/index.html`** — Minimal web UI for the legacy `/generate` endpoint.
- **`output/`** — Directory where generated MP3 files are persisted.

## Requirements

- Python 3.12+
- An OpenRouter API key

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

| Variable                  | Default                              | Description                                  |
|---------------------------|--------------------------------------|----------------------------------------------|
| `OPENROUTER_API_KEY`      | —                                    | OpenRouter API key (required)                |
| `OPENROUTER_BASE_URL`     | `https://openrouter.ai/api/v1`       | OpenRouter endpoint                          |
| `MUSIC_AGENT_MODEL`       | `google/gemini-3.1-flash-lite`       | Agent (LLM orchestration) model              |
| `AGENT_MAX_TOKENS`        | `2000`                               | Max tokens for the structured song payload   |
| `MUSIC_OUTRO_MODEL`       | `google/gemini-3.1-flash-lite`       | Outro agent model                            |
| `MUSIC_OUTRO_MAX_TOKENS`  | `100`                                | Max tokens for the outro message             |
| `MUSIC_OUTRO_TIMEOUT_SECONDS` | `20`                             | Outbound HTTP timeout for the outro agent    |
| `LYRIA_MODEL`             | `google/lyria-3-clip-preview`        | Lyria music generation model                 |
| `REDIS_URL`               | `redis://localhost:6379/0`           | Session memory backend                       |
| `SESSION_TTL_SECONDS`     | `1800`                               | Session expiration (30 minutes)              |
| `OUTPUT_DIR`              | `output`                             | Generated audio output directory             |
| `LOG_LEVEL`               | `INFO`                               | Structured log level                         |
| `HTTP_TIMEOUT_SECONDS`    | `120`                                | Outbound HTTP timeout for Lyria              |
| `LLM_TIMEOUT_SECONDS`     | `60`                                 | Outbound HTTP timeout for the agent          |

## Running the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts at `http://127.0.0.1:8000`. If `REDIS_URL` is unreachable, the service logs a warning and uses in-memory session storage (state is lost on restart).

## API Reference

### `GET /`

Serves the interactive web UI (`static/index.html`).

---

### `POST /generate` (legacy, backward compatible)

Generates music directly from a raw text prompt (no agent). The prompt is sent straight to Lyria.

**Request Body** (form data):

| Field  | Type   | Required | Description                      |
|--------|--------|----------|----------------------------------|
| prompt | string | Yes      | Description of the music to generate |

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "prompt=relaxing piano music with a female voice"
```

**Success Response (200):**

```json
{
  "audio": "/audio"
}
```

---

### `POST /assistant/generate` (agent flow)

Runs the user's message through the autonomous Music Creation Agent. The agent ALWAYS turns the request directly into the final structured song payload and starts Lyria generation — it never asks follow-up questions. Missing details (genre, mood, lyrics, ...) are filled with reasonable creative decisions.

**Request Body** (JSON):

| Field       | Type   | Required | Description                                       |
|-------------|--------|----------|---------------------------------------------------|
| session_id  | string | No       | Stable id for the conversation. Generated if omitted. |
| message     | string | Yes      | User's request (e.g. voice-transcribed text)      |

**When generation starts (always):**

```json
{
  "status": "generating",
  "session_id": "uuid",
  "prompt": "Sing only the Persian lyrics. The pronunciation field is only a phonetic guide...\nStyle: ...\nLyrics: ...",
  "audio": "/audio/uuid"
}
```

The `prompt` is the full Lyria prompt rendered from the structured payload (instruction + style + lyrics with pronunciation). `audio` points to the generated MP3.

```bash
curl -X POST http://127.0.0.1:8000/assistant/generate \
  -H "Content-Type: application/json" \
  -d '{"session_id":"uuid","message":"یه آهنگ برای تولدم بساز"}'
```

**Error Response (502):**

```json
{
  "status": "error",
  "message": "مشکلی پیش اومد، دوباره امتحان کن.",
  "session_id": "uuid"
}
```

### Agent behavior

- The agent is a creation engine, NOT a conversational assistant. It never asks for a theme, lyrics or details, and it never responds with conversational guidance.
- Every request — even a bare `یه آهنگ بساز` or a request with only a theme/genre/mood — produces a complete payload immediately:
  - `instruction`: fixed guidance to the music model (sing only the Persian lyrics; pronunciation is a phonetic guide).
  - `style`: genre, mood, vocal style, emotion and performance directions.
  - `lyrics`: array of `{persian, pronunciation}` lines with native Persian pronunciation guides (spoken contractions, `aa`/`oo`/`ee`/`kh`/`gh`/`ch`/`sh`/`zh`, regional accents preserved).
- If the model returns plain text, questions or invalid JSON, the agent retries once with a correction prompt before failing.
- The conversation history is never replayed into a new generation — each request starts fresh with the system instruction and the current user request only.
- Session state (generated payload, message history) expires after 30 minutes.

---

### `POST /assistant/outro` (post-playback outro)

Called after the user finishes listening to a generated song. The service loads the session, retrieves the stored generation context (metadata, original request, optimized Lyria prompt, conversation history) and asks the **outro agent** for a short, natural spoken Persian message. Only the text to be voiced by Galatea is returned — Lyria is never called and the generated audio is never modified.

**Request Body** (JSON):

| Field       | Type   | Required | Description                                          |
|-------------|--------|----------|------------------------------------------------------|
| session_id  | string | Yes      | Session id of the finished music generation          |
| event       | string | No       | Playback event (default `playback_finished`)         |

```bash
curl -X POST http://127.0.0.1:8000/assistant/outro \
  -H "Content-Type: application/json" \
  -d '{"session_id":"uuid","event":"playback_finished"}'
```

**Success Response (200):**

```json
{
  "status": "success",
  "message": "این آهنگ یه حس آرام و احساسی داشت. دوست داری نسخه‌ای با حال‌وهوای متفاوت هم امتحان کنیم؟"
}
```

**Behavior:**

- If the session is missing/expired or the LLM call fails, the safe fallback is returned so Galatea always has something to speak:

  `امیدوارم از این آهنگ لذت برده باشی. دوست داری یه آهنگ دیگه هم بسازیم؟`

- `400` when `session_id` is missing or invalid.

---

### `GET /audio`

Returns the most recently generated file from the legacy `/generate` endpoint.

### `GET /audio/{session_id}`

Returns the audio generated for a specific assistant session.

```bash
curl -o music.mp3 http://127.0.0.1:8000/audio/uuid
```

---

## Observability

All logs are structured JSON lines. Key events include `request_id`, `session_id`, model names, LLM latency, Lyria latency and generation status:

```json
{
  "time": "2026-08-26T11:39:33.201Z",
  "level": "INFO",
  "logger": "lyiar",
  "message": "assistant_generating",
  "request_id": "04265a739ada4f009bfa0f0c1cc84c05",
  "session_id": "test-session-1",
  "agent_model": "google/gemini-3.1-flash-lite",
  "lyria_model": "google/lyria-3-clip-preview",
  "llm_latency_ms": 1273,
  "lyria_latency_ms": 4349,
  "generation_status": "generating"
}
```

Pipeline events emitted by the `/assistant/generate` handler:

| Log message | When |
|---|---|
| `assistant_request_received` | a request arrived at `/assistant/generate` (with `session_id`, `message_chars`) |
| `music_agent_generated` | the autonomous agent produced the structured payload (with `lyrics_lines`, `llm_latency_ms`) |
| `assistant_generating` | Lyria generation started |
| `lyria_generation_started` | Lyria generation began (with `music_prompt_chars`) |
| `lyria_generation_failed` / `lyria_generation_success` | Lyria outcome |
| `music_agent_failed` | the agent/model call failed (502) |
| `outro_generation_started` | an outro request arrived (with `request_id`, `session_id`, `genre`, `mood`, `theme`) |
| `outro_generation_completed` | the outro was produced (with `llm_latency_ms`, `message_length`) |
| `outro_agent_failed` | the outro model call failed (fallback text is returned) |
| `outro_session_not_found` | the session expired/was missing (fallback text is returned) |

Every HTTP request also emits an `http_request` log with `request_id`, `method`, `path`, `status_code` and `duration_ms`. Set `LOG_LEVEL=DEBUG` for more detail.

## Agent internals

- Model: `google/gemini-3.1-flash-lite` (configurable via `MUSIC_AGENT_MODEL`).
- `temperature: 0.2`, `max_tokens: 2000` (configurable via `AGENT_MAX_TOKENS`), `response_format: {type: "json_object"}` for fast, deterministic JSON output.
- The agent is a creation engine and always replies with the final structured payload only:

  ```json
  {
    "instruction": "Sing only the Persian lyrics. The pronunciation field is only a phonetic guide. ...",
    "style": "Powerful male vocal, emotional gritty voice, cinematic atmosphere, energetic chorus.",
    "lyrics": [
      { "persian": "آروم آروم می‌آی", "pronunciation": "aroom aroom miyaa" },
      { "persian": "دلمَه که بی تو آروم نمی‌گیره", "pronunciation": "delema ke bi to aroom nemigire" }
    ]
  }
  ```

- The agent never asks for missing information: missing theme, genre, mood or lyrics are filled with creative decisions.
- If the model returns plain text, questions, explanations or invalid JSON (or a payload missing `instruction`/`style`/`lyrics`/`pronunciation`), the agent retries once with the correction prompt `"Return only the required JSON format. Do not explain. Do not ask questions."` before failing.
- The conversation history is never replayed into a new generation; each request is generated fresh from the system instruction + current user message.

## Outro agent internals

- Model: `google/gemini-3.1-flash-lite` (configurable via `MUSIC_OUTRO_MODEL`).
- `temperature: 0.4`, `max_tokens: 100` (configurable via `MUSIC_OUTRO_MAX_TOKENS`), `response_format: text`.
- The agent replies with plain spoken Persian only (1–3 sentences, max 35 words) — no JSON, no markdown, no explanations.
- Context sent to the model: song metadata, the structured song payload, original user request, the exact Lyria prompt that was generated, and the session conversation history (persisted by `/assistant/generate` at generation time).

## Deployment

### Docker Compose (recommended)

```bash
docker compose up -d --build
```

Starts the music service on port `8000` and a Redis instance for session memory. `OPENROUTER_API_KEY` is read from the shell environment (or `.env`):

```bash
OPENROUTER_API_KEY=sk-or-... docker compose up -d --build
```

### Docker

```bash
docker build -t lyiar-music-service .
docker run -d --name lyiar-music-service \
  -e OPENROUTER_API_KEY=sk-or-... \
  -p 8000:8000 \
  -v "$PWD/output:/app/output" \
  lyiar-music-service
```

Generated audio is persisted in the `output/` volume.

## Notes

- The agent flow generates the full Lyria prompt from the structured payload (instruction + style + lyrics with pronunciation); the legacy `/generate` endpoint sends the raw prompt to Lyria for backward compatibility.
- Each assistant session writes its audio to `output/{session_id}.mp3`; the legacy endpoint writes `output/lyria.mp3`.
- The Music Service is completely independent from any other project or repository.
