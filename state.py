import os
import json
import tempfile
import logging
from models import CommitInfo, LinearIssueRecord
import config

logger = logging.getLogger("state")


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
            return issue if issue.identifier == identifier else None
        return None
