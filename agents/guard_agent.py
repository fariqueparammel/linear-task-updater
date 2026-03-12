"""
Guard Agent — Spam & Duplicate Detection
Lightweight, zero-LLM-cost gatekeeper that intercepts CREATE_NEW actions
before they reach Linear. Checks for duplicate commits, similar titles,
rate limiting, and generic/noise titles.

Only blocks CREATE_NEW actions — ADD_SUBTASK and UPDATE_EXISTING always pass.
"""

import re
import logging
from difflib import SequenceMatcher
from datetime import datetime, timezone
from models import GeminiResult, LinearIssueRecord, GuardVerdict
from state import StateManager

logger = logging.getLogger("agent.guard")

# --- Thresholds ---
TITLE_SIMILARITY_THRESHOLD = 0.82
TITLE_SIMILARITY_WINDOW = 86400      # Compare against issues from last 24 hours
RATE_LIMIT_MAX_ISSUES = 5
RATE_LIMIT_WINDOW = 3600             # 1 hour
MIN_TITLE_LENGTH = 8

# Generic/noise title patterns — too vague to be meaningful issues
GENERIC_TITLE_PATTERNS = re.compile(
    r"^("
    r"update(d)?\s*(code|files?|project|app|stuff)?"
    r"|fix(ed)?\s*(bug|issue|error|it|stuff|things?)?"
    r"|changes?"
    r"|minor\s*(changes?|updates?|fixes?)?"
    r"|wip"
    r"|work\s*in\s*progress"
    r"|misc(ellaneous)?"
    r"|refactor(ing)?"
    r"|cleanup"
    r"|clean\s*up"
    r"|improvements?"
    r"|tweaks?"
    r"|adjustments?"
    r"|modifications?"
    r"|small\s*(changes?|fixes?|updates?)?"
    r"|various\s*(changes?|fixes?|updates?)?"
    r"|initial\s*commit"
    r"|first\s*commit"
    r"|test(ing)?"
    r"|temp(orary)?"
    r"|todo"
    r"|no\s*message"
    r"|untitled"
    r")$",
    re.IGNORECASE,
)


class GuardAgent:
    """
    Lightweight spam/duplicate detection agent.
    Runs ZERO LLM calls — uses string matching and rate checks only.
    Only evaluates CREATE_NEW actions; other actions pass unconditionally.
    """

    def __init__(self, state: StateManager):
        self._state = state
        logger.info("GuardAgent initialized (zero-LLM-cost spam/duplicate gate)")

    def evaluate(
        self,
        result: GeminiResult,
        source_shas: list[str],
        created_issues: list[LinearIssueRecord],
        user_recent_issues: list[dict] | None = None,
        primary_author: str | None = None,
    ) -> GuardVerdict:
        """
        Evaluate whether a Gemini classification should proceed to Linear.

        Only blocks CREATE_NEW actions. ADD_SUBTASK and UPDATE_EXISTING
        always pass since they reference existing issues.

        Returns GuardVerdict with allowed=True/False and reason.
        """
        if result.action != "CREATE_NEW":
            return GuardVerdict(allowed=True, reason="", check_name="passed")

        checks = [
            ("sha_duplicate", lambda: self._check_duplicate_sha(source_shas, created_issues)),
            ("title_similarity", lambda: self._check_title_similarity(result.title, created_issues, user_recent_issues)),
            ("rate_limit", lambda: self._check_rate_limit(primary_author, created_issues)),
            ("generic_title", lambda: self._check_generic_title(result.title)),
        ]

        for check_name, check_fn in checks:
            reason = check_fn()
            if reason:
                logger.warning(
                    f"GUARD_BLOCKED | check={check_name} | "
                    f"reason={reason} | title=\"{result.title}\" | "
                    f"author={primary_author or 'unknown'}"
                )
                return GuardVerdict(allowed=False, reason=reason, check_name=check_name)

        logger.debug(
            f"GUARD_PASSED | title=\"{result.title}\" | author={primary_author or 'unknown'}"
        )
        return GuardVerdict(allowed=True, reason="", check_name="passed")

    # --- Check 1: Duplicate SHA ---

    def _check_duplicate_sha(
        self,
        source_shas: list[str],
        created_issues: list[LinearIssueRecord],
    ) -> str | None:
        """Block if any commit SHA was already used to create an issue."""
        if not source_shas:
            return None

        known_shas: set[str] = set()
        for issue in created_issues:
            known_shas.update(issue.source_commits)

        duplicates = [sha for sha in source_shas if sha in known_shas]
        if duplicates:
            short_shas = ", ".join(s[:8] for s in duplicates[:3])
            return f"Commit(s) already processed: {short_shas}"

        return None

    # --- Check 2: Title Similarity ---

    def _check_title_similarity(
        self,
        title: str,
        created_issues: list[LinearIssueRecord],
        user_recent_issues: list[dict] | None = None,
    ) -> str | None:
        """Block if the proposed title is too similar to a recently created issue."""
        if not title:
            return None

        normalized = self._normalize_title(title)
        now = datetime.now(timezone.utc)

        # Compare against script-owned issues from the last 24 hours
        for issue in created_issues:
            try:
                created_dt = datetime.fromisoformat(issue.created_at)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age = (now - created_dt).total_seconds()
                if age > TITLE_SIMILARITY_WINDOW:
                    continue
            except (ValueError, TypeError):
                continue

            existing_title = issue.title or self._extract_title_from_url(issue.url)
            if not existing_title:
                continue

            ratio = SequenceMatcher(
                None, normalized, self._normalize_title(existing_title)
            ).ratio()

            if ratio >= TITLE_SIMILARITY_THRESHOLD:
                return (
                    f"Similar to existing {issue.identifier} "
                    f"(similarity={ratio:.2f}, existing=\"{existing_title[:60]}\")"
                )

        # Compare against user's recent Linear issues (have actual titles)
        if user_recent_issues:
            for recent in user_recent_issues:
                recent_title = recent.get("title", "")
                if not recent_title:
                    continue

                ratio = SequenceMatcher(
                    None, normalized, self._normalize_title(recent_title)
                ).ratio()

                if ratio >= TITLE_SIMILARITY_THRESHOLD:
                    identifier = recent.get("identifier", "?")
                    return (
                        f"Similar to recent {identifier} "
                        f"(similarity={ratio:.2f}, existing=\"{recent_title[:60]}\")"
                    )

        return None

    # --- Check 3: Rate Limit ---

    def _check_rate_limit(
        self,
        author: str | None,
        created_issues: list[LinearIssueRecord],
    ) -> str | None:
        """Block if too many issues created for the same author recently."""
        if not author:
            return None

        now = datetime.now(timezone.utc)
        recent_count = 0

        for issue in created_issues:
            if issue.commit_author != author:
                continue
            try:
                created_dt = datetime.fromisoformat(issue.created_at)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age = (now - created_dt).total_seconds()
                if age <= RATE_LIMIT_WINDOW:
                    recent_count += 1
            except (ValueError, TypeError):
                continue

        if recent_count >= RATE_LIMIT_MAX_ISSUES:
            return (
                f"Rate limit: author '{author}' has {recent_count} issues "
                f"in the last {RATE_LIMIT_WINDOW // 60} min (max {RATE_LIMIT_MAX_ISSUES})"
            )

        return None

    # --- Check 4: Generic Title ---

    def _check_generic_title(self, title: str) -> str | None:
        """Block if the title is too generic or vague to be meaningful."""
        if not title:
            return "Empty title"

        stripped = title.strip()

        if len(stripped) < MIN_TITLE_LENGTH:
            return f"Title too short ({len(stripped)} chars, min {MIN_TITLE_LENGTH})"

        if GENERIC_TITLE_PATTERNS.match(stripped):
            return f"Generic/noise title: \"{stripped}\""

        return None

    # --- Helpers ---

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize for comparison: lowercase, strip punctuation, collapse whitespace."""
        title = title.lower().strip()
        title = re.sub(r"[^\w\s]", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    @staticmethod
    def _extract_title_from_url(url: str) -> str | None:
        """
        Extract title from Linear issue URL slug.
        URL format: https://linear.app/<workspace>/issue/<ID>/<slug>
        """
        if not url:
            return None
        parts = url.rstrip("/").split("/")
        if len(parts) < 2:
            return None
        slug = parts[-1]
        if re.match(r"^[A-Z]+-\d+$", slug, re.IGNORECASE):
            return None
        return slug.replace("-", " ")
