"""
Gemini Agent
Classifies commit batches using Gemini AI with 3-key rotation.
Determines whether to create, update, or add a subtask in Linear.
"""

import json
import time
import logging
from itertools import cycle
import requests
from models import CommitInfo, GeminiResult, LinearIssueRecord
import config

logger = logging.getLogger("agent.gemini")

SYSTEM_PROMPT = """\
You are a project management AI. You analyze git commit messages and decide \
what Linear (project management) action to take.

Respond with ONLY a valid JSON object, no markdown fences, no explanation.

{
  "action": "CREATE_NEW" | "ADD_SUBTASK" | "UPDATE_EXISTING",
  "title": "concise task title (max 80 chars)",
  "description": "bullet-point summary of the commits",
  "priority": 0-4,
  "label": "Bug" | "Feature" | "Improvement" | "Chore" | "Refactor",
  "state": "Todo" | "In Progress" | "Done",
  "parent_issue_id": "TEAM-123 or null",
  "existing_issue_id": "TEAM-456 or null"
}

Priority scale: 0=No priority, 1=Urgent, 2=High, 3=Medium, 4=Low.

Decision rules:
- CREATE_NEW: Commits introduce new work not tied to any existing tracked task.
- ADD_SUBTASK: Commits are clearly a sub-part of a larger tracked task. You MUST \
set parent_issue_id to one of the known issue identifiers listed below.
- UPDATE_EXISTING: Commits continue work on an already tracked task. You MUST \
set existing_issue_id to one of the known issue identifiers listed below.
- If uncertain, default to CREATE_NEW.
- For ADD_SUBTASK and UPDATE_EXISTING, you may ONLY reference identifiers from \
the list of script-created issues provided below. If no match fits, use CREATE_NEW.
"""

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class KeyManager:
    """Rotates through Gemini API keys."""

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._cycle = cycle(keys)
        self._current = next(self._cycle)

    def get_key(self) -> str:
        return self._current

    def rotate(self) -> str:
        self._current = next(self._cycle)
        logger.info("Rotated to next Gemini API key")
        return self._current


class GeminiAgent:
    """Classifies commit batches using Gemini AI."""

    def __init__(self, keys: list[str]):
        self._key_manager = KeyManager(keys)

    def classify(
        self,
        commits: list[CommitInfo],
        created_issues: list[LinearIssueRecord],
    ) -> GeminiResult | None:
        """
        Send a commit batch to Gemini and get a classification result.
        Returns None if all retries fail.
        """
        user_prompt = self._build_user_prompt(commits, created_issues)
        logger.info(f"Classifying batch of {len(commits)} commit(s) with Gemini")

        for attempt in range(MAX_RETRIES):
            try:
                result = self._call_gemini(user_prompt)
                if result:
                    errors = result.validate()
                    if errors:
                        logger.warning(f"Gemini result validation errors: {errors}")
                        # Fall back to CREATE_NEW if action is invalid
                        if result.action not in GeminiResult.VALID_ACTIONS:
                            result.action = "CREATE_NEW"
                    logger.info(
                        f"Gemini classified as {result.action}: {result.title}"
                    )
                    return result
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    self._key_manager.rotate()
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Gemini 429 rate limit. Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini API error: {e}")
                    break
            except Exception as e:
                logger.error(f"Gemini call failed: {e}")
                break

        logger.error("All Gemini retries exhausted. Skipping batch.")
        return None

    def _call_gemini(self, user_prompt: str) -> GeminiResult | None:
        """Make a single Gemini API call and parse the JSON response."""
        url = f"{config.GEMINI_API_BASE}?key={self._key_manager.get_key()}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT},
                        {"text": user_prompt},
                    ]
                }
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
            logger.warning("Empty response from Gemini")
            return None

        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            parsed = json.loads(text)
            return GeminiResult.from_dict(parsed)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini JSON: {e}\nRaw: {text[:500]}")
            return None

    def _build_user_prompt(
        self,
        commits: list[CommitInfo],
        created_issues: list[LinearIssueRecord],
    ) -> str:
        """Format commits and known issues into the user prompt."""
        lines = ["Recent commits to classify:\n"]
        for i, c in enumerate(commits, 1):
            lines.append(
                f"Commit {i} (by {c.author}, repo: {c.repo}, {c.timestamp}):\n{c.message}\n"
            )

        if created_issues:
            lines.append("\nKnown script-created Linear issues:")
            # Show the most recent 20 issues for context
            for issue in created_issues[-20:]:
                lines.append(f"  - {issue.identifier}: {issue.url}")
        else:
            lines.append("\nNo existing script-created issues yet.")

        return "\n".join(lines)
