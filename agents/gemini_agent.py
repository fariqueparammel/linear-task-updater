"""
Gemini Agent
Classifies commit batches using Gemini AI with key + model rotation.
Rotates through all (key, model) combinations to maximize free-tier throughput.
Example: 3 keys × 5 models = 15 unique slots before any repeat.
"""

import json
import time
import logging
from itertools import product, cycle
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
  "existing_issue_id": "TEAM-456 or null",
  "is_critical": true | false
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
the list of script-created issues OR the commit author's recent tasks listed below.
- Set is_critical to true ONLY if the commit is a hotfix, security patch, \
crash fix, or addresses a breaking/urgent production issue. Otherwise false.

Per-user correlation:
- You will also receive the commit author's recent Linear tasks.
- If the commit message is clearly related to one of the author's recent tasks \
(same feature area, same component, continuation of work), prefer ADD_SUBTASK \
with that task as parent, or UPDATE_EXISTING if the task was script-created.
- If the commit is unrelated to any of the author's recent tasks, use CREATE_NEW.
"""

MAX_RETRIES = 8
RETRY_BASE_DELAY = 3  # seconds
MAX_RETRY_DELAY = 60  # cap backoff at 60s


class SlotManager:
    """
    Rotates through all (api_key, model) combinations.
    With 3 keys × 5 models = 15 unique slots, each with its own rate limit.
    """

    def __init__(self, keys: list[str], models: list[str]):
        # Build all unique (key, model) pairs
        self._slots = list(product(keys, models))
        self._cycle = cycle(self._slots)
        self._current = next(self._cycle)
        self._total = len(self._slots)
        logger.info(
            f"SlotManager initialized: {len(keys)} key(s) × {len(models)} model(s) "
            f"= {self._total} rotation slot(s)"
        )

    @property
    def total_slots(self) -> int:
        return self._total

    def get(self) -> tuple[str, str]:
        """Returns current (api_key, model_name)."""
        return self._current

    def rotate(self) -> tuple[str, str]:
        """Advance to next (key, model) combination."""
        self._current = next(self._cycle)
        key, model = self._current
        logger.info(f"Rotated to: {model} (key ...{key[-4:]})")
        return self._current


class GeminiAgent:
    """Classifies commit batches using Gemini AI with key+model rotation."""

    def __init__(self, keys: list[str]):
        self._slot_manager = SlotManager(keys, config.GEMINI_MODELS)

    def classify(
        self,
        commits: list[CommitInfo],
        created_issues: list[LinearIssueRecord],
        user_recent_issues: list[dict] | None = None,
    ) -> GeminiResult | None:
        """
        Send a commit batch to Gemini and get a classification result.
        Returns None if all retries fail.
        """
        user_prompt = self._build_user_prompt(commits, created_issues, user_recent_issues)
        logger.info(f"Classifying batch of {len(commits)} commit(s) with Gemini")

        for attempt in range(MAX_RETRIES):
            try:
                result = self._call_gemini(user_prompt)
                if result:
                    errors = result.validate()
                    if errors:
                        logger.warning(f"Gemini result validation errors: {errors}")
                        if result.action not in GeminiResult.VALID_ACTIONS:
                            result.action = "CREATE_NEW"
                    key, model = self._slot_manager.get()
                    logger.info(
                        f"Gemini ({model}) classified as {result.action}: {result.title}"
                    )
                    return result
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    self._slot_manager.rotate()
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    logger.warning(
                        f"429 rate limit. Rotating slot, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini API error: {e}")
                    # Try next slot for non-429 errors too (model might not support JSON mode etc.)
                    self._slot_manager.rotate()
            except Exception as e:
                logger.error(f"Gemini call failed: {e}")
                self._slot_manager.rotate()

        logger.error("All Gemini retries exhausted. Skipping batch.")
        return None

    def _call_gemini(self, user_prompt: str) -> GeminiResult | None:
        """Make a single Gemini API call using the current (key, model) slot."""
        key, model = self._slot_manager.get()
        url = f"{config.GEMINI_API_BASE}/{model}:generateContent?key={key}"

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
            logger.warning(f"Empty response from {model}")
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
            logger.warning(f"Failed to parse JSON from {model}: {e}\nRaw: {text[:500]}")
            return None

    def _build_user_prompt(
        self,
        commits: list[CommitInfo],
        created_issues: list[LinearIssueRecord],
        user_recent_issues: list[dict] | None = None,
    ) -> str:
        """Format commits, known issues, and user's recent tasks into the user prompt."""
        lines = ["Recent commits to classify:\n"]
        for i, c in enumerate(commits, 1):
            lines.append(
                f"Commit {i} (by {c.author}, repo: {c.repo}, {c.timestamp}):\n{c.message}\n"
            )

        if created_issues:
            lines.append("\nKnown script-created Linear issues:")
            for issue in created_issues[-20:]:
                lines.append(f"  - {issue.identifier}: {issue.url}")
        else:
            lines.append("\nNo existing script-created issues yet.")

        if user_recent_issues:
            # Deduce author from first commit
            author = commits[0].author if commits else "unknown"
            lines.append(f"\nCommit author ({author})'s recent Linear tasks:")
            for issue in user_recent_issues:
                lines.append(
                    f"  - {issue['identifier']}: \"{issue['title']}\" ({issue['state']})"
                )
            lines.append(
                "\nIf the commit is related to one of the author's tasks above, "
                "prefer ADD_SUBTASK or UPDATE_EXISTING referencing that task."
            )
        else:
            lines.append("\nNo recent tasks found for the commit author.")

        return "\n".join(lines)
