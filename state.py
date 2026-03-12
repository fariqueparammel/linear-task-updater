import os
import json
import tempfile
import time
import logging
from datetime import datetime, timezone
from models import CommitInfo, LinearIssueRecord
import config

logger = logging.getLogger("state")

# Cache TTL defaults (in seconds)
USER_ISSUES_TTL = 600       # 10 minutes — user's recent issues
TEAM_MEMBERS_TTL = 3600     # 1 hour — team member mapping
STALE_CLEANUP_INTERVAL = 300  # 5 minutes — how often to auto-purge
LLM_CLEANUP_INTERVAL = 3600  # 1 hour — how often to run LLM-driven cleanup


class StateManager:
    """Manages persistent state across restarts using atomic file writes."""

    def __init__(self):
        os.makedirs(config.STATE_DIR, exist_ok=True)
        self._repo_shas_path = os.path.join(config.STATE_DIR, "repo_shas.json")
        self._buffer_path = os.path.join(config.STATE_DIR, "commit_buffer.json")
        self._issues_path = os.path.join(config.STATE_DIR, "created_issues.json")

    # --- Atomic file I/O ---

    def _read_json(self, path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Corrupt state file {path}, resetting: {e}")
            return default

    def _write_json(self, path, data):
        """Write JSON atomically: write to temp file, then rename."""
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # --- Per-repo SHA tracking ---

    def get_repo_shas(self) -> dict[str, str]:
        """Returns {repo_full_name: last_processed_sha}."""
        return self._read_json(self._repo_shas_path, {})

    def update_repo_shas(self, updated: dict[str, str]):
        """Merge updated SHAs into existing state."""
        current = self.get_repo_shas()
        current.update(updated)
        self._write_json(self._repo_shas_path, current)
        logger.debug(f"Updated SHAs for {len(updated)} repos")

    # --- Commit buffer ---

    def load_buffer(self) -> list[CommitInfo]:
        raw = self._read_json(self._buffer_path, [])
        return [CommitInfo.from_dict(c) for c in raw]

    def save_buffer(self, buffer: list[CommitInfo]):
        self._write_json(self._buffer_path, [c.to_dict() for c in buffer])

    # --- Created issues registry ---

    def get_created_issues(self) -> list[LinearIssueRecord]:
        raw = self._read_json(self._issues_path, [])
        return [LinearIssueRecord.from_dict(r) for r in raw]

    def add_created_issue(self, record: LinearIssueRecord):
        issues = self.get_created_issues()
        issues.append(record)
        self._write_json(self._issues_path, [i.to_dict() for i in issues])
        logger.info(f"Registered created issue: {record.identifier}")

    def is_owned_issue(self, identifier: str) -> bool:
        """Check if an issue identifier (e.g. TEAM-123) was created by this script."""
        return any(i.identifier == identifier for i in self.get_created_issues())

    def get_issue_by_identifier(self, identifier: str) -> LinearIssueRecord | None:
        for issue in self.get_created_issues():
            if issue.identifier == identifier:
                return issue
        return None


class CacheManager:
    """
    JSON-based cache with TTL for expensive API lookups.
    Stores entries as {key: {"data": ..., "ts": unix_timestamp}}.
    Auto-purges stale entries on read/write to keep the file clean.
    """

    def __init__(self, state_dir: str | None = None):
        cache_dir = state_dir or config.STATE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_path = os.path.join(cache_dir, "cache.json")
        self._last_cleanup = 0.0
        self._last_llm_cleanup = 0.0

    def _load(self) -> dict:
        if not os.path.exists(self._cache_path):
            return {}
        try:
            with open(self._cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Corrupt cache file, resetting: {e}")
            return {}

    def _save(self, data: dict):
        dir_name = os.path.dirname(self._cache_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get(self, key: str, ttl: int) -> dict | list | None:
        """
        Retrieve a cached value if it exists and is not older than ttl seconds.
        Returns None on miss or expiry.
        """
        self._maybe_cleanup()
        cache = self._load()
        entry = cache.get(key)
        if entry is None:
            return None

        age = time.time() - entry.get("ts", 0)
        if age > ttl:
            logger.debug(f"Cache expired for '{key}' (age={age:.0f}s > ttl={ttl}s)")
            return None

        logger.debug(f"Cache hit for '{key}' (age={age:.0f}s)")
        return entry["data"]

    def set(self, key: str, data):
        """Store a value in the cache with the current timestamp."""
        cache = self._load()
        cache[key] = {"data": data, "ts": time.time()}
        self._save(cache)
        logger.debug(f"Cache set for '{key}'")

    def invalidate(self, key: str):
        """Remove a specific key from the cache."""
        cache = self._load()
        if key in cache:
            del cache[key]
            self._save(cache)
            logger.debug(f"Cache invalidated for '{key}'")

    def purge_stale(self, max_age: int = 86400):
        """Remove all entries older than max_age seconds (default 24h)."""
        cache = self._load()
        now = time.time()
        before = len(cache)
        cache = {
            k: v for k, v in cache.items()
            if now - v.get("ts", 0) <= max_age
        }
        removed = before - len(cache)
        if removed > 0:
            self._save(cache)
            logger.info(f"Cache purge: removed {removed} stale entries (>{max_age}s old)")
        return removed

    def _maybe_cleanup(self):
        """Auto-purge very old entries periodically."""
        now = time.time()
        if now - self._last_cleanup < STALE_CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        self.purge_stale(max_age=86400)  # Remove entries older than 24h

    def stats(self) -> dict:
        """Return cache statistics for debugging."""
        cache = self._load()
        now = time.time()
        total = len(cache)
        if total == 0:
            return {"total": 0, "oldest_age": 0, "newest_age": 0}
        ages = [now - v.get("ts", 0) for v in cache.values()]
        return {
            "total": total,
            "oldest_age": int(max(ages)),
            "newest_age": int(min(ages)),
        }

    def get_entries_summary(self) -> list[dict]:
        """Return a summary of all cache entries for LLM-driven cleanup."""
        cache = self._load()
        now = time.time()
        entries = []
        for key, val in cache.items():
            age_seconds = int(now - val.get("ts", 0))
            data = val.get("data")
            # Provide a size/type hint without dumping full data
            if isinstance(data, list):
                data_hint = f"list with {len(data)} items"
            elif isinstance(data, dict):
                data_hint = f"dict with {len(data)} keys"
            else:
                data_hint = type(data).__name__
            entries.append({
                "key": key,
                "age_seconds": age_seconds,
                "age_human": _format_age(age_seconds),
                "data_type": data_hint,
            })
        return entries

    def should_run_llm_cleanup(self) -> bool:
        """Check if enough time has passed for an LLM-driven cache review."""
        now = time.time()
        if now - self._last_llm_cleanup < LLM_CLEANUP_INTERVAL:
            return False
        # Only worth running if there are entries to review
        cache = self._load()
        return len(cache) > 0

    def mark_llm_cleanup_done(self):
        """Record that LLM cleanup was just performed."""
        self._last_llm_cleanup = time.time()

    def remove_keys(self, keys: list[str]) -> int:
        """Remove specific keys from the cache. Returns count removed."""
        cache = self._load()
        removed = 0
        for key in keys:
            if key in cache:
                del cache[key]
                removed += 1
        if removed > 0:
            self._save(cache)
            logger.info(f"LLM cache cleanup: removed {removed} entries: {keys}")
        return removed


def _format_age(seconds: int) -> str:
    """Format seconds into a human-readable age string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    elif seconds < 86400:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"
