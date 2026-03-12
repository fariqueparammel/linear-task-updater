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
  "label": "<one of the valid labels listed below>",
  "state": "<one of the valid workflow states listed below>",
  "assignee": "<exact displayName of the team member to assign this to, or null>",
  "project": "<one of the active project names listed below, or null>",
  "parent_issue_id": "TEAM-123 or null",
  "existing_issue_id": "TEAM-456 or null",
  "is_critical": true | false
}

=== FIELD RULES ===

Priority (REQUIRED — integer 0-4):
  0 = No priority, 1 = Urgent, 2 = High, 3 = Medium, 4 = Low.
  Choose based on commit impact: hotfixes/security → 1, features → 3, chores → 4.

Label (REQUIRED — must be from the valid labels list):
  Pick the label that best categorizes the work. If none fit perfectly, pick the \
closest match. Do NOT invent labels.

State (REQUIRED — must be from the valid workflow states list):
  For new tasks from commits, use an in-progress or to-do state. \
Do NOT use "Done" unless the commit explicitly indicates completion.

Assignee (REQUIRED — must be an exact team member displayName):
  You MUST assign the task to the person who made the commit.
  The commit provides the GitHub username. The team member list provides Linear \
display names and emails. Match the GitHub username to the correct team member \
using name similarity, email prefix, or process of elimination.
  The assignee value MUST be the EXACT displayName from the team member list. \
Do NOT invent names. If you cannot determine the correct assignee, set to null.
  Use elimination: if only one team member's name/email could plausibly match \
the GitHub username, pick that person.

Project (OPTIONAL — pick from the active projects list if applicable):
  If the commit work clearly relates to one of the active projects, set the \
project name. Otherwise set to null. Do NOT invent project names.

=== DECISION RULES ===
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

=== PER-USER CORRELATION ===
- You will receive ONLY the commit author's own tasks (both script-created and \
their recent Linear tasks). You will NOT see other users' tasks.
- For ADD_SUBTASK or UPDATE_EXISTING, you may ONLY reference task identifiers \
from the lists provided below — these are exclusively the commit author's tasks.
- If the commit is clearly related to one of the author's existing tasks \
(same feature area, same component, continuation of work), prefer ADD_SUBTASK \
with that task as parent, or UPDATE_EXISTING if the task was script-created.
- If the commit is unrelated to any of the author's tasks, use CREATE_NEW.
- NEVER reference a task identifier that is not in the provided lists — it may \
belong to another user.
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
        self._improvement_agent = None  # Set via set_improvement_agent()
        self._context_additions = ""    # Active learned rules appended to prompt

    def set_improvement_agent(self, improvement_agent):
        """Inject the ImprovementAgent for self-improving prompts."""
        self._improvement_agent = improvement_agent
        self._context_additions = improvement_agent.get_active_context_additions()
        if self._context_additions:
            logger.info(f"Loaded {len(improvement_agent.get_active_improvements())} learned rules into prompt")

    def classify(
        self,
        commits: list[CommitInfo],
        created_issues: list[LinearIssueRecord],
        user_recent_issues: list[dict] | None = None,
        workspace_context: dict | None = None,
    ) -> GeminiResult | None:
        """
        Send a commit batch to Gemini and get a classification result.
        Returns None if all retries fail.
        """
        user_prompt = self._build_user_prompt(
            commits, created_issues, user_recent_issues, workspace_context
        )
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

        # Combine base prompt with any learned rules from ImprovementAgent
        full_system_prompt = SYSTEM_PROMPT + self._context_additions

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_system_prompt},
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
        workspace_context: dict | None = None,
    ) -> str:
        """Format commits, known issues, workspace context, and user's recent tasks."""
        lines = ["Recent commits to classify:\n"]
        for i, c in enumerate(commits, 1):
            lines.append(
                f"Commit {i} (by {c.author}, repo: {c.repo}, {c.timestamp}):\n{c.message}\n"
            )

        # Workspace metadata — valid values for Gemini to pick from
        if workspace_context:
            members = workspace_context.get("members", [])
            if members:
                lines.append("\n=== TEAM MEMBERS (use EXACT displayName for assignee) ===")
                for m in members:
                    email_part = f" ({m['email']})" if m.get("email") else ""
                    lines.append(f"  - {m['displayName']}{email_part}")
                # Remind about matching
                author = commits[0].author if commits else "unknown"
                lines.append(
                    f"\nThe commit author's GitHub username is \"{author}\". "
                    f"Match this to one of the team members above. "
                    f"Use name similarity, email prefix, or elimination to determine "
                    f"the correct assignee. The assignee MUST be an exact displayName from above."
                )

            states = workspace_context.get("workflow_states", [])
            if states:
                lines.append(f"\n=== VALID WORKFLOW STATES (use one of these for \"state\") ===")
                lines.append(f"  {', '.join(states)}")

            labels = workspace_context.get("labels", [])
            if labels:
                lines.append(f"\n=== VALID LABELS (use one of these for \"label\") ===")
                lines.append(f"  {', '.join(labels)}")

            projects = workspace_context.get("projects", [])
            if projects:
                lines.append(f"\n=== ACTIVE PROJECTS (use one for \"project\", or null) ===")
                lines.append(f"  {', '.join(projects)}")

            cycles = workspace_context.get("cycles", [])
            if cycles:
                active_note = " (auto-assigned to current cycle)" if workspace_context.get("has_active_cycle") else ""
                lines.append(f"\n=== CYCLES{active_note} ===")
                lines.append(f"  {', '.join(cycles)}")

        # Script-created issues (filtered to this author only)
        author = commits[0].author if commits else "unknown"
        if created_issues:
            lines.append(f"\nScript-created issues by {author} (author's own tasks only):")
            for issue in created_issues[-20:]:
                title_part = f" \"{issue.title}\"" if issue.title else ""
                lines.append(f"  - {issue.identifier}:{title_part} {issue.url}")
        else:
            lines.append(f"\nNo script-created issues by {author} yet.")

        # User's recent tasks for correlation
        if user_recent_issues:
            lines.append(f"\n{author}'s recent Linear tasks (author's own tasks only):")
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

    def evaluate_cache_cleanup(self, entries: list[dict]) -> list[str]:
        """
        Ask Gemini which cache entries should be purged.
        Returns list of cache keys to remove.
        """
        if not entries:
            return []

        prompt = self._build_cache_cleanup_prompt(entries)

        try:
            key, model = self._slot_manager.get()
            url = f"{config.GEMINI_API_BASE}/{model}:generateContent?key={key}"

            system = (
                "You are a cache management AI. You analyze cached data entries and decide "
                "which ones are stale and should be removed to keep the cache efficient.\n\n"
                "Respond with ONLY a valid JSON object:\n"
                '{"remove": ["key1", "key2"], "reason": "brief explanation"}\n\n'
                "Rules:\n"
                "- KEEP entries that are frequently refreshed (user_issues:*, team_members*) "
                "if they are less than 2 hours old — they will be re-fetched naturally.\n"
                "- REMOVE entries older than 6 hours — they are likely stale.\n"
                "- REMOVE user_issues:* entries older than 30 minutes — user context changes fast.\n"
                "- KEEP team_members and team_members_detail if under 2 hours — expensive to re-fetch.\n"
                "- If all entries are fresh and useful, return {\"remove\": [], \"reason\": \"all entries are fresh\"}.\n"
                "- Be conservative: when in doubt, keep the entry."
            )

            payload = {
                "contents": [{"parts": [{"text": system}, {"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                },
            }

            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()

            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            if not text:
                return []

            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

            parsed = json.loads(text.strip())
            keys_to_remove = parsed.get("remove", [])
            reason = parsed.get("reason", "no reason given")

            if keys_to_remove:
                logger.info(f"LLM cache review: removing {len(keys_to_remove)} entries — {reason}")
            else:
                logger.debug(f"LLM cache review: no entries to remove — {reason}")

            return keys_to_remove

        except Exception as e:
            logger.warning(f"LLM cache cleanup failed (non-critical): {e}")
            return []

    @staticmethod
    def _build_cache_cleanup_prompt(entries: list[dict]) -> str:
        """Format cache entries for the LLM to review."""
        lines = [
            f"Current cache has {len(entries)} entries. Review each and decide which to remove:\n"
        ]
        for e in entries:
            lines.append(
                f"  - key=\"{e['key']}\" | age={e['age_human']} ({e['age_seconds']}s) | type={e['data_type']}"
            )
        lines.append(
            "\nReturn the keys that should be removed as a JSON array in the 'remove' field."
        )
        return "\n".join(lines)
