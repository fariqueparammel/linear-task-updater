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
- You will see ALL team members' tasks (script-created and recent), each labeled \
with their owner. The commit author is clearly identified.
- For ADD_SUBTASK or UPDATE_EXISTING, you may ONLY target issues that belong to \
the commit author (marked "← YOURS" in the issue list). NEVER target another \
user's task — the system will reject it.
- If the commit is clearly related to one of the author's existing tasks \
(same feature area, same component, continuation of work), prefer ADD_SUBTASK \
with that task as parent, or UPDATE_EXISTING if the task was script-created.
- If another team member already has a similar task but it does NOT belong to \
the commit author, still use CREATE_NEW — each user manages their own tasks.
- If the commit is unrelated to any of the author's tasks, use CREATE_NEW.

=== LARGE CHANGES ===
- Commits may include file/line change stats (e.g. "10 files changed, 800 lines").
- For commits with many files changed (10+) or many lines (500+), these represent \
significant work. If the author already has a related parent task, use ADD_SUBTASK \
to break it down. If no related task exists, use CREATE_NEW.
- Large commits deserve higher priority (2 or above) and a detailed description \
summarizing what the changes cover.
"""

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
        self._keys = keys
        self._slot_manager = SlotManager(keys, config.GEMINI_MODELS)
        self._improvement_agent = None  # Set via set_improvement_agent()
        self._context_additions = ""    # Active learned rules appended to prompt
        self._active_models = list(config.GEMINI_MODELS)  # Track discovered models

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
        primary_author: str | None = None,
    ) -> GeminiResult | None:
        """
        Send a commit batch to Gemini and get a classification result.
        Retries across ALL available (key, model) slots before giving up.
        Returns None only if every slot has been exhausted.
        """
        user_prompt = self._build_user_prompt(
            commits, created_issues, user_recent_issues, workspace_context,
            primary_author
        )
        total_slots = self._slot_manager.total_slots
        # Try every slot at least once, plus a few extra retries for transient errors
        max_attempts = total_slots + 3
        logger.info(
            f"CLASSIFY_START | batch={len(commits)} commit(s) | "
            f"slots={total_slots} | max_attempts={max_attempts}"
        )

        failed_models = set()
        for attempt in range(max_attempts):
            key, model = self._slot_manager.get()
            try:
                result = self._call_gemini(user_prompt)
                if result:
                    errors = result.validate()
                    if errors:
                        logger.warning(f"Gemini result validation errors: {errors}")
                        if result.action not in GeminiResult.VALID_ACTIONS:
                            result.action = "CREATE_NEW"
                    logger.info(
                        f"CLASSIFY_SUCCESS | model={model} | key=...{key[-4:]} | "
                        f"action={result.action} | title={result.title} | "
                        f"attempt={attempt + 1}/{max_attempts}"
                    )
                    return result
                else:
                    # Empty response — model may not support JSON mode
                    logger.warning(
                        f"CLASSIFY_EMPTY | model={model} | key=...{key[-4:]} | "
                        f"attempt={attempt + 1}/{max_attempts} — rotating"
                    )
                    failed_models.add(model)
                    self._slot_manager.rotate()

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    delay = min(RETRY_BASE_DELAY * (2 ** min(attempt, 5)), MAX_RETRY_DELAY)
                    logger.warning(
                        f"CLASSIFY_RATE_LIMITED | model={model} | key=...{key[-4:]} | "
                        f"HTTP 429 — rotating slot, retry in {delay}s "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    self._slot_manager.rotate()
                    time.sleep(delay)
                elif status in (400, 404):
                    # Model doesn't exist or doesn't support this request
                    logger.warning(
                        f"CLASSIFY_MODEL_ERROR | model={model} | HTTP {status} — "
                        f"marking as failed, rotating"
                    )
                    failed_models.add(model)
                    self._slot_manager.rotate()
                else:
                    logger.error(
                        f"CLASSIFY_API_ERROR | model={model} | HTTP {status}: {e} — rotating"
                    )
                    self._slot_manager.rotate()
            except Exception as e:
                logger.error(
                    f"CLASSIFY_FAILED | model={model} | {e} — rotating "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                self._slot_manager.rotate()

        logger.error(
            f"CLASSIFY_EXHAUSTED | All {max_attempts} attempts failed. "
            f"Failed models: {sorted(failed_models) if failed_models else 'none'}. "
            f"Skipping batch."
        )
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
        primary_author: str | None = None,
    ) -> str:
        """Format commits, known issues, workspace context, and user's recent tasks."""
        author = primary_author or (commits[0].author if commits else "unknown")
        lines = ["Recent commits to classify:\n"]
        for i, c in enumerate(commits, 1):
            stats_part = ""
            if c.files_changed or c.lines_changed:
                stats_part = f", {c.files_changed} files changed, {c.lines_changed} lines"
            lines.append(
                f"Commit {i} (by {c.author}, repo: {c.repo}, {c.timestamp}{stats_part}):\n{c.message}\n"
            )

        # Workspace metadata — valid values for Gemini to pick from
        if workspace_context:
            members = workspace_context.get("members", [])
            if members:
                lines.append("\n=== TEAM MEMBERS (use EXACT displayName for assignee) ===")
                for m in members:
                    email_part = f" ({m['email']})" if m.get("email") else ""
                    lines.append(f"  - {m['displayName']}{email_part}")
                lines.append(
                    f"\nThe commit author's GitHub username is \"{author}\". "
                    f"Match this to one of the team members above. "
                    f"GitHub usernames often combine first+last name (e.g. 'johndoe' = 'John Doe', "
                    f"'fariqueparammel' = 'farique'). Check if the username contains or starts with "
                    f"a team member's displayName or email prefix. "
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

        # Script-created issues — ALL team members, labeled by owner
        if created_issues:
            lines.append(f"\n=== SCRIPT-CREATED ISSUES (all team members) ===")
            lines.append(f"Commit author: {author}")
            for issue in created_issues[-30:]:
                title_part = f" \"{issue.title}\"" if issue.title else ""
                owner_tag = f" [owner: {issue.commit_author}]" if issue.commit_author else " [owner: unknown]"
                is_mine = " ← YOURS" if issue.commit_author == author else ""
                lines.append(f"  - {issue.identifier}:{title_part}{owner_tag}{is_mine} {issue.url}")
            lines.append(
                f"\nFor UPDATE_EXISTING or ADD_SUBTASK, you may ONLY target issues "
                f"owned by the commit author ({author}), marked with '← YOURS' above."
            )
        else:
            lines.append("\nNo script-created issues yet.")

        # User's recent tasks for correlation
        if user_recent_issues:
            lines.append(f"\n=== {author.upper()}'s RECENT LINEAR TASKS ===")
            for issue in user_recent_issues:
                lines.append(
                    f"  - {issue['identifier']}: \"{issue['title']}\" ({issue['state']})"
                )
            lines.append(
                f"\nIf the commit is related to one of {author}'s tasks above, "
                "prefer ADD_SUBTASK or UPDATE_EXISTING referencing that task."
            )
        else:
            lines.append(f"\nNo recent tasks found for {author}.")

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

    def discover_and_refresh_models(self) -> list[str]:
        """
        Query the Gemini models.list API to discover available text-to-text models.
        Filters for generateContent-capable models and rebuilds the slot rotation.
        Returns list of discovered model names, or empty list on failure.
        """
        key = self._keys[0] if self._keys else None
        if not key:
            logger.warning("MODEL_DISCOVERY | No API keys available")
            return []

        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=100"

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            models = data.get("models", [])
            text_models = []
            skipped_models = []

            for m in models:
                name = m.get("name", "")  # e.g. "models/gemini-2.5-flash"
                short_name = name.replace("models/", "")
                supported = m.get("supportedGenerationMethods", [])
                input_limit = m.get("inputTokenLimit", 0)
                output_limit = m.get("outputTokenLimit", 0)
                description = m.get("description", "").lower()

                # Must support generateContent (text generation)
                if "generateContent" not in supported:
                    skipped_models.append(f"{short_name} (no generateContent)")
                    continue

                # Must have both input and output token limits (text-in → text-out)
                if not input_limit or not output_limit:
                    skipped_models.append(f"{short_name} (no token limits)")
                    continue

                # Skip non-text models: embedding, vision-only, image gen, video gen, audio, AQA
                NON_TEXT_MARKERS = (
                    "embedding", "aqa", "imagen", "veo", "music",
                    "tts", "speech", "whisper", "moderation",
                )
                if any(skip in short_name for skip in NON_TEXT_MARKERS):
                    skipped_models.append(f"{short_name} (non-text model)")
                    continue

                # Skip if description indicates non-text-generation purpose
                NON_TEXT_DESCRIPTIONS = ("embed", "image generation", "video generation", "text-to-speech")
                if any(nd in description for nd in NON_TEXT_DESCRIPTIONS):
                    skipped_models.append(f"{short_name} (non-text description)")
                    continue

                # Must have a reasonable output limit for JSON classification responses
                if output_limit < 256:
                    skipped_models.append(f"{short_name} (output limit too small: {output_limit})")
                    continue

                text_models.append(short_name)
                # Log model capabilities
                logger.info(
                    f"MODEL_AVAILABLE | {short_name} | "
                    f"input={input_limit} | output={output_limit} | "
                    f"methods={','.join(supported)}"
                )

            logger.info(
                f"MODEL_DISCOVERY | API returned {len(models)} models total — "
                f"{len(text_models)} text-to-text, {len(skipped_models)} skipped"
            )
            if skipped_models:
                logger.debug(f"MODEL_DISCOVERY | Skipped: {', '.join(skipped_models)}")

            if not text_models:
                logger.warning("MODEL_DISCOVERY | No text-to-text models found from API")
                return []

            # Verify each model actually works with a quick test call
            total_candidates = len(text_models)
            verified_models = []
            for model_name in text_models:
                if self._verify_model(key, model_name):
                    verified_models.append(model_name)
                else:
                    logger.warning(f"MODEL_VERIFY_FAILED | {model_name} — excluded from rotation")

            if not verified_models:
                logger.warning(
                    "MODEL_DISCOVERY | No models passed verification. "
                    "Keeping current model list unchanged."
                )
                return self._active_models

            text_models = verified_models
            logger.info(
                f"MODEL_VERIFY | {len(verified_models)}/{total_candidates} "
                f"models passed verification"
            )

            # Compare with current list
            current_set = set(self._active_models)
            discovered_set = set(text_models)
            new_models = discovered_set - current_set
            removed_models = current_set - discovered_set

            if new_models:
                logger.info(f"MODEL_DISCOVERY | New models available: {sorted(new_models)}")
            if removed_models:
                logger.info(f"MODEL_DISCOVERY | Models no longer available: {sorted(removed_models)}")

            if new_models or removed_models:
                self._active_models = text_models
                # Rebuild slot manager with updated model list
                self._slot_manager = SlotManager(self._keys, text_models)
                logger.info(
                    f"MODEL_REFRESH | Rebuilt slot rotation with {len(text_models)} models: "
                    f"{', '.join(text_models)}"
                )
            else:
                logger.info(
                    f"MODEL_DISCOVERY | No changes. {len(text_models)} models still available."
                )

            return text_models

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            logger.warning(f"MODEL_DISCOVERY | API error (HTTP {status}): {e}")
            return []
        except Exception as e:
            logger.warning(f"MODEL_DISCOVERY | Failed: {e}")
            return []

    def _verify_model(self, api_key: str, model_name: str) -> bool:
        """
        Quick verification that a model can produce JSON text output.
        Sends a tiny prompt and checks for a valid response.
        Returns True if the model works, False if it fails.
        """
        url = f"{config.GEMINI_API_BASE}/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": "Reply with exactly: {\"status\":\"ok\"}"}]}
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 32,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 429:
                # Rate limited but model exists — assume it works
                logger.debug(f"MODEL_VERIFY | {model_name}: 429 (rate limited, assuming OK)")
                return True
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if text.strip():
                logger.debug(f"MODEL_VERIFY | {model_name}: OK")
                return True
            else:
                logger.debug(f"MODEL_VERIFY | {model_name}: empty response")
                return False
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            logger.debug(f"MODEL_VERIFY | {model_name}: HTTP {status}")
            return False
        except Exception as e:
            logger.debug(f"MODEL_VERIFY | {model_name}: {e}")
            return False

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
