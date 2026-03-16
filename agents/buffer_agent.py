"""
Buffer Agent
Manages commit batching: collects commits and triggers processing
when batch size is reached or timeout expires.
Critical commits (hotfix, security, crash) bypass the batch threshold.
"""

import re
import logging
from datetime import datetime, timezone
from models import CommitInfo
from state import StateManager
import config

logger = logging.getLogger("agent.buffer")

CRITICAL_PATTERNS = re.compile(
    r"(hotfix|critical|urgent|security|CVE-|crash|breaking|emergency|rollback)",
    re.IGNORECASE,
)

# Thresholds for "large change" bypass — commits with major changes shouldn't wait
LARGE_CHANGE_FILES_THRESHOLD = 10     # 10+ files changed
LARGE_CHANGE_LINES_THRESHOLD = 500    # 500+ lines changed


class BufferAgent:
    """Manages commit buffering and batch-ready detection."""

    def __init__(self, state: StateManager):
        self._state = state

    def add_commits(self, commits: list[CommitInfo]):
        """Append new commits to the buffer."""
        buffer = self._state.load_buffer()
        buffer.extend(commits)
        self._state.save_buffer(buffer)
        logger.info(f"Buffered {len(commits)} commit(s). Buffer size: {len(buffer)}")

    def has_critical(self) -> bool:
        """Check if any buffered commit matches critical keywords."""
        buffer = self._state.load_buffer()
        for commit in buffer:
            if CRITICAL_PATTERNS.search(commit.message):
                logger.info(f"Critical commit detected (keyword): {commit.sha[:8]} — {commit.message[:60]}")
                return True
        return False

    def has_large_change(self) -> bool:
        """Check if any buffered commit has massive file/line changes."""
        buffer = self._state.load_buffer()
        for commit in buffer:
            if commit.files_changed >= LARGE_CHANGE_FILES_THRESHOLD:
                logger.info(
                    f"Large change detected (files={commit.files_changed}): "
                    f"{commit.sha[:8]} — {commit.message[:60]}"
                )
                return True
            if commit.lines_changed >= LARGE_CHANGE_LINES_THRESHOLD:
                logger.info(
                    f"Large change detected (lines={commit.lines_changed}): "
                    f"{commit.sha[:8]} — {commit.message[:60]}"
                )
                return True
        return False

    def is_ready(self) -> bool:
        """
        Returns True if a batch should be processed:
        - Buffer has >= BATCH_SIZE commits, OR
        - Buffer is non-empty and contains a critical commit, OR
        - Buffer is non-empty and the oldest commit is older than BATCH_TIMEOUT_SECONDS.
        """
        buffer = self._state.load_buffer()
        if not buffer:
            return False

        if len(buffer) >= config.BATCH_SIZE:
            return True

        # Critical commits bypass batch threshold
        if self.has_critical():
            logger.info(
                f"Critical commit in buffer — bypassing batch threshold. "
                f"Processing {len(buffer)} commit(s) immediately."
            )
            return True

        # Large changes (many files or lines) bypass batch threshold
        if self.has_large_change():
            logger.info(
                f"Large change in buffer — bypassing batch threshold. "
                f"Processing {len(buffer)} commit(s) immediately."
            )
            return True

        # Check timeout on oldest commit
        oldest_ts = buffer[0].timestamp
        try:
            oldest_dt = datetime.fromisoformat(oldest_ts)
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - oldest_dt).total_seconds()
            if age >= config.BATCH_TIMEOUT_SECONDS:
                logger.info(
                    f"Batch timeout reached ({age:.0f}s). "
                    f"Force-processing {len(buffer)} commit(s)."
                )
                return True
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse timestamp '{oldest_ts}': {e}")

        return False

    def get_batch(self) -> list[CommitInfo]:
        """
        Pop up to BATCH_SIZE commits from a single author.
        Per-author batching ensures each Gemini call handles one user's work,
        so commits from different users across different repos are never mixed.
        """
        buffer = self._state.load_buffer()
        if not buffer:
            return []

        # Take the first commit's author and batch only their commits
        target_author = buffer[0].author
        author_commits = [c for c in buffer if c.author == target_author]
        batch = author_commits[: config.BATCH_SIZE]

        if len(set(c.author for c in buffer[:config.BATCH_SIZE])) > 1:
            logger.info(
                f"PER_AUTHOR_BATCH | Batching {len(batch)} commit(s) for author={target_author} "
                f"(other authors queued separately)"
            )

        return batch

    def pending_count(self) -> int:
        """Return number of commits currently buffered."""
        return len(self._state.load_buffer())

    def clear_batch(self, batch: list[CommitInfo]):
        """Remove processed commits from the buffer."""
        buffer = self._state.load_buffer()
        processed_shas = {c.sha for c in batch}
        remaining = [c for c in buffer if c.sha not in processed_shas]
        self._state.save_buffer(remaining)
        logger.debug(f"Cleared {len(batch)} from buffer. Remaining: {len(remaining)}")
