import json
import logging
import threading
import time

logger = logging.getLogger("lyiar.session_store")

try:
    import redis as _redis
    REDIS_AVAILABLE = True
except ImportError:
    _redis = None
    REDIS_AVAILABLE = False


class SessionStore:
    def __init__(self, redis_url, ttl_seconds):
        self.ttl_seconds = ttl_seconds
        self._redis = None
        self._memory = {}
        self._lock = threading.Lock()

        if REDIS_AVAILABLE:
            try:
                client = _redis.from_url(
                    redis_url,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
                client.ping()
                self._redis = client
                logger.info("session_store_backend", extra={"backend": "redis"})
            except Exception as exc:
                logger.warning(
                    "session_store_redis_unavailable_falling_back_to_memory",
                    extra={"backend": "memory", "error": str(exc)},
                )
        else:
            logger.warning(
                "session_store_redis_package_missing_falling_back_to_memory",
                extra={"backend": "memory"},
            )

    @property
    def backend(self):
        return "redis" if self._redis is not None else "memory"

    def get(self, session_id):
        if self._redis is not None:
            try:
                raw = self._redis.get(self._key(session_id))
            except Exception as exc:
                logger.warning(
                    "session_store_redis_get_failed",
                    extra={"session_id": session_id, "error": str(exc)},
                )
                return None
            if not raw:
                return None
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return None

        with self._lock:
            entry = self._memory.get(session_id)
            if not entry:
                return None
            if time.time() - entry.get("saved_at", 0) > self.ttl_seconds:
                self._memory.pop(session_id, None)
                return None
            return entry.get("data")

    def set(self, session_id, data):
        payload = json.dumps(data, ensure_ascii=False)

        if self._redis is not None:
            try:
                self._redis.set(self._key(session_id), payload, ex=self.ttl_seconds)
            except Exception as exc:
                logger.warning(
                    "session_store_redis_set_failed",
                    extra={"session_id": session_id, "error": str(exc)},
                )
            return

        with self._lock:
            self._memory[session_id] = {
                "saved_at": time.time(),
                "data": data,
            }

    def delete(self, session_id):
        if self._redis is not None:
            try:
                self._redis.delete(self._key(session_id))
            except Exception as exc:
                logger.warning(
                    "session_store_redis_delete_failed",
                    extra={"session_id": session_id, "error": str(exc)},
                )
            return

        with self._lock:
            self._memory.pop(session_id, None)

    @staticmethod
    def _key(session_id):
        return "music_assistant:session:{}".format(session_id)
