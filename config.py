import os


def _get_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
).rstrip("/")

MUSIC_AGENT_MODEL = os.getenv(
    "MUSIC_AGENT_MODEL",
    "google/gemini-3.1-flash-lite"
)
LYRIA_MODEL = os.getenv(
    "LYRIA_MODEL",
    "google/lyria-3-clip-preview"
)

# Outro generation (post-playback conversation).
MUSIC_OUTRO_MODEL = os.getenv(
    "MUSIC_OUTRO_MODEL",
    "google/gemini-3.1-flash-lite"
)

AGENT_TEMPERATURE = 0.2
AGENT_MAX_TOKENS = _get_int("AGENT_MAX_TOKENS", 2000)

# Outro agent settings: low temperature keeps the spoken message natural,
# the token cap keeps it short (voice-friendly).
MUSIC_OUTRO_TEMPERATURE = 0.4
MUSIC_OUTRO_MAX_TOKENS = _get_int("MUSIC_OUTRO_MAX_TOKENS", 100)
MUSIC_OUTRO_TIMEOUT_SECONDS = _get_int("MUSIC_OUTRO_TIMEOUT_SECONDS", 20)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = _get_int("SESSION_TTL_SECONDS", 1800)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

HTTP_TIMEOUT_SECONDS = _get_int("HTTP_TIMEOUT_SECONDS", 120)
LLM_TIMEOUT_SECONDS = _get_int("LLM_TIMEOUT_SECONDS", 60)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
