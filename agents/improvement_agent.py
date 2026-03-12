"""
Improvement Agent — Self-Improving Context System

Tracks classification accuracy and iteratively improves the Gemini prompt context.
An improved context is only promoted to "active" after it has been validated against
at least 20 newly created tasks with 100% correct field classification.

The current (baseline) context is NEVER deleted — improvements are additive.
If an improved context fails validation, it's discarded and the baseline is kept.

Stores all state in the persistent JSON cache so it survives restarts.
"""

import json
import time
import logging
import requests
from state import CacheManager
import config

logger = logging.getLogger("agent.improvement")

# How many consecutive perfect classifications before promoting
PROMOTION_THRESHOLD = 20
# How often to attempt generating an improvement (seconds)
IMPROVEMENT_INTERVAL = 7200  # Every 2 hours
# Cache key prefixes
TRACKER_KEY = "improvement:tracker"
CANDIDATE_KEY = "improvement:candidate"
HISTORY_KEY = "improvement:history"

IMPROVEMENT_PROMPT = """\
You are a prompt engineering AI. You analyze the performance of a classification \
system and suggest improvements to its system prompt.

You will receive:
1. The current system prompt used for classifying git commits into Linear tasks
2. Recent classification results with their accuracy (which fields were correct/incorrect)
3. Common error patterns

Your task: suggest specific, targeted improvements to the system prompt that would \
fix the observed errors WITHOUT breaking what already works correctly.

Respond with ONLY a valid JSON object:
{
  "improvements": [
    {
      "section": "which section to modify (e.g. 'Priority rules', 'Label rules')",
      "current_text": "the problematic text or rule (brief excerpt)",
      "suggested_text": "the improved text or rule",
      "reasoning": "why this change would fix the observed errors"
    }
  ],
  "summary": "one-sentence summary of all improvements"
}

Rules:
- Be CONSERVATIVE. Small, targeted changes only.
- Do NOT rewrite the entire prompt. Only fix specific rules that caused errors.
- Do NOT remove existing rules — only refine or add clarifying rules.
- If the system is performing well (>90% accuracy), return {"improvements": [], "summary": "no changes needed"}.
- Focus on the most frequent error patterns first.
- Each improvement should be independently testable.
"""


class ImprovementAgent:
    """
    Self-improving agent that tracks classification accuracy and suggests
    prompt improvements. Only promotes improvements after PROMOTION_THRESHOLD
    consecutive perfect classifications.
    """

    def __init__(self, gemini_keys: list[str], cache: CacheManager):
        self._keys = gemini_keys
        self._key_index = 0
        self._cache = cache
        self._last_improvement_attempt = 0.0

    def _get_next_key(self) -> str:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    # --- Tracking ---

    def record_classification(
        self,
        task_identifier: str,
        gemini_result: dict,
        actual_fields: dict | None = None,
        was_correct: bool = True,
        error_details: str | None = None,
    ):
        """
        Record a classification result for accuracy tracking.

        Args:
            task_identifier: Linear issue identifier (e.g. "LAT-876")
            gemini_result: The raw GeminiResult fields
            actual_fields: What the fields should have been (if known)
            was_correct: Whether all fields were correctly assigned
            error_details: Description of what went wrong (if any)
        """
        tracker = self._load_tracker()

        entry = {
            "task": task_identifier,
            "timestamp": time.time(),
            "fields": {
                "priority": gemini_result.get("priority"),
                "label": gemini_result.get("label"),
                "state": gemini_result.get("state"),
                "assignee": gemini_result.get("assignee"),
                "project": gemini_result.get("project"),
                "action": gemini_result.get("action"),
            },
            "correct": was_correct,
            "error": error_details,
        }

        tracker["classifications"].append(entry)
        # Keep last 100 entries
        tracker["classifications"] = tracker["classifications"][-100:]
        tracker["total_count"] += 1
        if was_correct:
            tracker["correct_count"] += 1
            tracker["consecutive_correct"] += 1
        else:
            tracker["consecutive_correct"] = 0
            tracker["error_patterns"].append({
                "task": task_identifier,
                "error": error_details or "unknown",
                "fields": entry["fields"],
                "timestamp": time.time(),
            })
            tracker["error_patterns"] = tracker["error_patterns"][-50:]

        self._save_tracker(tracker)

        accuracy = (
            tracker["correct_count"] / tracker["total_count"] * 100
            if tracker["total_count"] > 0 else 0
        )
        logger.debug(
            f"Classification recorded: {task_identifier} "
            f"(correct={was_correct}, streak={tracker['consecutive_correct']}, "
            f"accuracy={accuracy:.1f}%)"
        )

        # Check if candidate context should be promoted
        candidate = self._load_candidate()
        if candidate and tracker["consecutive_correct"] >= PROMOTION_THRESHOLD:
            self._promote_candidate(candidate, tracker)

    def _load_tracker(self) -> dict:
        """Load or initialize the accuracy tracker."""
        data = self._cache.get(TRACKER_KEY, ttl=86400 * 365)  # Never expire
        if data:
            return data
        return {
            "classifications": [],
            "total_count": 0,
            "correct_count": 0,
            "consecutive_correct": 0,
            "error_patterns": [],
            "active_context_version": 0,
        }

    def _save_tracker(self, tracker: dict):
        self._cache.set(TRACKER_KEY, tracker)

    def _load_candidate(self) -> dict | None:
        return self._cache.get(CANDIDATE_KEY, ttl=86400 * 7)  # 7 day TTL

    def _save_candidate(self, candidate: dict):
        self._cache.set(CANDIDATE_KEY, candidate)

    # --- Promotion ---

    def _promote_candidate(self, candidate: dict, tracker: dict):
        """
        Promote a candidate context improvement to active status after
        it has been validated with PROMOTION_THRESHOLD consecutive correct
        classifications.
        """
        version = tracker.get("active_context_version", 0) + 1

        # Save to history (never delete old versions)
        history = self._cache.get(HISTORY_KEY, ttl=86400 * 365) or []
        history.append({
            "version": version,
            "promoted_at": time.time(),
            "improvements": candidate.get("improvements", []),
            "summary": candidate.get("summary", ""),
            "validation_streak": tracker["consecutive_correct"],
            "accuracy_at_promotion": (
                tracker["correct_count"] / tracker["total_count"] * 100
                if tracker["total_count"] > 0 else 0
            ),
        })
        self._cache.set(HISTORY_KEY, history)

        # Update tracker
        tracker["active_context_version"] = version
        tracker["consecutive_correct"] = 0  # Reset streak
        self._save_tracker(tracker)

        # Clear candidate
        self._cache.invalidate(CANDIDATE_KEY)

        logger.info(
            f"CONTEXT PROMOTED | Version {version} | "
            f"Validated with {PROMOTION_THRESHOLD}+ consecutive correct | "
            f"Summary: {candidate.get('summary', 'N/A')}"
        )

    # --- Improvement Generation ---

    def should_attempt_improvement(self) -> bool:
        """Check if enough time and data has passed to try generating an improvement."""
        now = time.time()
        if now - self._last_improvement_attempt < IMPROVEMENT_INTERVAL:
            return False

        tracker = self._load_tracker()
        # Need at least 10 classifications before attempting improvements
        if tracker["total_count"] < 10:
            return False

        # Don't attempt if already have an active candidate being validated
        candidate = self._load_candidate()
        if candidate:
            return False

        # Only attempt if there are recent errors
        if not tracker["error_patterns"]:
            return False

        return True

    def generate_improvement(self, current_system_prompt: str) -> dict | None:
        """
        Analyze recent classification results and generate a prompt improvement.
        Returns the improvement suggestion or None.
        """
        self._last_improvement_attempt = time.time()

        tracker = self._load_tracker()
        if not tracker["error_patterns"]:
            logger.debug("No error patterns to improve on")
            return None

        accuracy = (
            tracker["correct_count"] / tracker["total_count"] * 100
            if tracker["total_count"] > 0 else 0
        )

        # If accuracy is very high, no improvements needed
        if accuracy >= 95 and len(tracker["error_patterns"]) < 3:
            logger.info(
                f"IMPROVEMENT SKIP | Accuracy is {accuracy:.1f}% with "
                f"{len(tracker['error_patterns'])} errors — no improvement needed"
            )
            return None

        prompt = self._build_improvement_prompt(
            current_system_prompt, tracker
        )

        try:
            result = self._call_gemini(prompt)
            if result and result.get("improvements"):
                # Save as candidate (NOT active yet — needs validation)
                candidate = {
                    "improvements": result["improvements"],
                    "summary": result.get("summary", ""),
                    "generated_at": time.time(),
                    "accuracy_at_generation": accuracy,
                    "error_count_at_generation": len(tracker["error_patterns"]),
                }
                self._save_candidate(candidate)

                logger.info(
                    f"IMPROVEMENT CANDIDATE GENERATED | "
                    f"{len(result['improvements'])} suggestion(s) | "
                    f"Summary: {result.get('summary', 'N/A')} | "
                    f"Needs {PROMOTION_THRESHOLD} consecutive correct to promote"
                )
                return result
            else:
                logger.info("IMPROVEMENT | Gemini found no improvements needed")
                return None

        except Exception as e:
            logger.warning(f"Improvement generation failed (non-critical): {e}")
            return None

    def get_active_improvements(self) -> list[dict]:
        """
        Return the list of all promoted (validated) improvements.
        These should be appended to the system prompt as additional rules.
        """
        history = self._cache.get(HISTORY_KEY, ttl=86400 * 365) or []
        all_improvements = []
        for entry in history:
            all_improvements.extend(entry.get("improvements", []))
        return all_improvements

    def get_active_context_additions(self) -> str:
        """
        Return additional prompt text from promoted improvements.
        This is appended to the base system prompt.
        """
        improvements = self.get_active_improvements()
        if not improvements:
            return ""

        lines = ["\n\n=== LEARNED RULES (validated through usage) ==="]
        for imp in improvements:
            section = imp.get("section", "General")
            text = imp.get("suggested_text", "")
            if text:
                lines.append(f"- [{section}] {text}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return improvement agent statistics."""
        tracker = self._load_tracker()
        history = self._cache.get(HISTORY_KEY, ttl=86400 * 365) or []
        candidate = self._load_candidate()

        accuracy = (
            tracker["correct_count"] / tracker["total_count"] * 100
            if tracker["total_count"] > 0 else 0
        )

        return {
            "total_classifications": tracker["total_count"],
            "correct_classifications": tracker["correct_count"],
            "accuracy_pct": round(accuracy, 1),
            "consecutive_correct": tracker["consecutive_correct"],
            "error_patterns": len(tracker["error_patterns"]),
            "promoted_versions": len(history),
            "has_candidate": candidate is not None,
            "promotion_threshold": PROMOTION_THRESHOLD,
        }

    # --- Gemini API ---

    def _call_gemini(self, user_prompt: str) -> dict | None:
        key = self._get_next_key()
        model = "gemini-2.5-flash-lite"
        url = f"{config.GEMINI_API_BASE}/{model}:generateContent?key={key}"

        payload = {
            "contents": [
                {"parts": [{"text": IMPROVEMENT_PROMPT}, {"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not text:
            return None

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse improvement response: {e}")
            return None

    def _build_improvement_prompt(
        self, current_prompt: str, tracker: dict
    ) -> str:
        lines = []

        accuracy = (
            tracker["correct_count"] / tracker["total_count"] * 100
            if tracker["total_count"] > 0 else 0
        )
        lines.append(
            f"Classification system stats: {tracker['total_count']} total, "
            f"{accuracy:.1f}% accuracy, "
            f"{len(tracker['error_patterns'])} recent errors\n"
        )

        lines.append("=== CURRENT SYSTEM PROMPT (DO NOT rewrite entirely) ===")
        # Only include the first 2000 chars to avoid token limits
        lines.append(current_prompt[:2000])
        if len(current_prompt) > 2000:
            lines.append("... [truncated]")

        if tracker["error_patterns"]:
            lines.append("\n=== RECENT ERRORS (fix these) ===")
            for err in tracker["error_patterns"][-10:]:
                fields = err.get("fields", {})
                lines.append(
                    f"  - Task {err.get('task', '?')}: {err.get('error', 'unknown')} "
                    f"| fields: priority={fields.get('priority')}, "
                    f"label={fields.get('label')}, state={fields.get('state')}, "
                    f"assignee={fields.get('assignee')}"
                )

        # Recent correct classifications for context
        recent_correct = [
            c for c in tracker["classifications"][-20:]
            if c.get("correct")
        ]
        if recent_correct:
            lines.append(f"\n=== RECENT CORRECT (keep these working) ===")
            for c in recent_correct[-5:]:
                fields = c.get("fields", {})
                lines.append(
                    f"  - Task {c.get('task', '?')}: "
                    f"priority={fields.get('priority')}, "
                    f"label={fields.get('label')}, state={fields.get('state')}"
                )

        lines.append(
            "\nSuggest targeted improvements to fix the errors "
            "while keeping correct classifications working."
        )

        return "\n".join(lines)
