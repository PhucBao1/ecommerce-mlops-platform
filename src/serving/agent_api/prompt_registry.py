"""
Prompt Registry — manages versioned system prompts for the shopping agent.

Stores active prompt version in Redis so all agent_api instances read the same
version without restart. Supports atomic rollback.

Usage:
    registry = PromptRegistry()
    registry.set_active("v2")        # deploy new version
    active = registry.get_active()   # → "v2"
    registry.rollback()              # → "v1" (previous version)

Prompt files: src/serving/agent_api/prompts/system_{version}.txt
"""

import logging
import os
import pathlib

import redis

logger = logging.getLogger(__name__)

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
_ACTIVE_KEY = "agent:prompt:active"
_HISTORY_KEY = "agent:prompt:history"
_HISTORY_MAX = 10


def _get_redis() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", "123"),
        decode_responses=True,
    )


class PromptRegistry:
    def __init__(self):
        self._r = _get_redis()

    def list_versions(self) -> list[str]:
        """Return sorted list of available prompt file versions."""
        versions = []
        for f in sorted(_PROMPTS_DIR.glob("system_*.txt")):
            versions.append(f.stem.replace("system_", ""))
        return versions

    def get_active(self) -> str:
        """Return the currently active prompt version (default: v1)."""
        return self._r.get(_ACTIVE_KEY) or os.getenv("AGENT_PROMPT_VERSION", "v1")

    def set_active(self, version: str) -> None:
        """Set a new active prompt version. Pushes previous to history stack."""
        if not (_PROMPTS_DIR / f"system_{version}.txt").exists():
            raise FileNotFoundError(f"Prompt file system_{version}.txt not found")
        current = self.get_active()
        if current:
            self._r.lpush(_HISTORY_KEY, current)
            self._r.ltrim(_HISTORY_KEY, 0, _HISTORY_MAX - 1)
        self._r.set(_ACTIVE_KEY, version)
        logger.info(
            "prompt_registry: active version set to %s (was %s)", version, current
        )

    def rollback(self) -> str:
        """Roll back to the previous prompt version. Returns the version rolled back to."""
        previous = self._r.lpop(_HISTORY_KEY)
        if not previous:
            logger.warning("prompt_registry: no previous version to rollback to")
            return self.get_active()
        self._r.set(_ACTIVE_KEY, previous)
        logger.info("prompt_registry: rolled back to %s", previous)
        return previous

    def get_prompt_text(self, version: str | None = None) -> str:
        """Load prompt text for a version (defaults to active version)."""
        version = version or self.get_active()
        path = _PROMPTS_DIR / f"system_{version}.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("prompt_registry: file not found for version %s", version)
            return ""

    def history(self) -> list[str]:
        """Return version history stack (most recent first)."""
        return self._r.lrange(_HISTORY_KEY, 0, _HISTORY_MAX - 1)
