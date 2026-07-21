"""
Policy Engine — YAML-based input guardrails for the shopping agent.

Loads rules from config/policies/input_rules.yaml and evaluates them
against every incoming request before it reaches the LLM.

Complements the existing Guardrails class (guardrails.py) which uses
hardcoded rules. PolicyEngine externalizes rules to YAML for:
  - No code changes needed to add/update rules
  - Ops team can tune policies without redeployment
  - Rules are versioned alongside config (git-tracked)

Usage:
    engine = PolicyEngine()
    result = engine.check("gợi ý tai nghe bluetooth")
    if not result.allowed:
        return blocked_response(result.reason)
"""

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = os.getenv(
    "POLICY_RULES_PATH",
    str(Path(__file__).parents[3] / "config" / "policies" / "input_rules.yaml"),
)


@dataclass
class PolicyResult:
    allowed: bool
    action: str  # "allow" | "warn" | "block"
    reason: str  # rule id or empty string
    rule_id: str


@lru_cache(maxsize=1)
def _load_rules(policy_path: str) -> list[dict]:
    """Load and cache YAML rules. Cached so file is read once per process."""
    try:
        with open(policy_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        rules = data.get("rules", [])
        logger.info("policy_engine loaded %d rules from %s", len(rules), policy_path)
        return rules
    except FileNotFoundError:
        logger.warning(
            "policy_engine: rules file not found at %s, no rules applied", policy_path
        )
        return []
    except yaml.YAMLError as e:
        logger.error("policy_engine: YAML parse error: %s", e)
        return []


class PolicyEngine:
    """Evaluate input text against YAML-defined policy rules."""

    def __init__(self, policy_path: str = _DEFAULT_POLICY_PATH):
        self._policy_path = policy_path
        self._rules = _load_rules(policy_path)

    def _match_rule(self, rule: dict, text: str) -> bool:
        """Return True if the rule matches the input text."""
        # Length checks
        min_len = rule.get("min_length")
        max_len = rule.get("max_length")
        if min_len is not None and len(text) < min_len:
            return True
        if max_len is not None and len(text) > max_len:
            return True

        # Keyword matching (exact substring, case-insensitive)
        for kw in rule.get("keywords", []):
            if re.search(kw, text, re.IGNORECASE | re.UNICODE):
                return True

        # Regex pattern matching
        for pattern in rule.get("patterns", []):
            if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
                return True

        return False

    def check(self, text: str) -> PolicyResult:
        """
        Evaluate text against all rules.

        Returns PolicyResult with action=block/warn/allow and the matching rule_id.
        First matching rule wins (rules evaluated in order from YAML).
        """
        if not text:
            return PolicyResult(
                allowed=False, action="block", reason="empty_input", rule_id="empty"
            )

        for rule in self._rules:
            if self._match_rule(rule, text):
                action = rule.get("action", "block")
                reason = rule.get("reason", rule.get("id", "unknown"))
                rule_id = rule.get("id", "unknown")

                if action == "block":
                    logger.info(
                        "policy_engine blocked rule=%s reason=%s text_len=%d",
                        rule_id,
                        reason,
                        len(text),
                    )
                    return PolicyResult(
                        allowed=False, action=action, reason=reason, rule_id=rule_id
                    )
                elif action == "warn":
                    logger.warning(
                        "policy_engine warn rule=%s reason=%s text_len=%d",
                        rule_id,
                        reason,
                        len(text),
                    )
                    return PolicyResult(
                        allowed=True, action=action, reason=reason, rule_id=rule_id
                    )

        return PolicyResult(allowed=True, action="allow", reason="", rule_id="")

    def reload(self) -> None:
        """Force reload of YAML rules (clears lru_cache)."""
        _load_rules.cache_clear()
        self._rules = _load_rules(self._policy_path)
        logger.info("policy_engine reloaded %d rules", len(self._rules))
