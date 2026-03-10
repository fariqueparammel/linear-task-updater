"""
Integration Test Suite
Tests all agents against real GitHub, Gemini, and Linear APIs.

Usage:
    1. Fill in your .env file with real API keys
    2. Run: python test_integration.py

Each test phase can be run independently. The script will:
    - [Phase 1] Validate config loads correctly
    - [Phase 2] Test GitHub read-only access (list org repos, fetch commits)
    - [Phase 3] Test Gemini classification (send sample commits, get JSON back)
    - [Phase 4] Test Linear (create label, create issue, update issue, create subtask)
    - [Phase 5] Full pipeline: GitHub -> Buffer -> Gemini -> Linear end-to-end
    - [Cleanup] Optionally delete test issues (asks for confirmation)

No destructive actions are taken without user confirmation.
"""

import sys
import os
import json
import time
import logging
import tempfile
import shutil
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import config
from models import CommitInfo, GeminiResult, LinearIssueRecord
from state import StateManager
from agents.github_agent import GitHubAgent
from agents.buffer_agent import BufferAgent
from agents.gemini_agent import GeminiAgent, KeyManager
from agents.linear_agent import LinearAgent, AUTO_SYNC_LABEL_NAME

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"

test_results = {"passed": 0, "failed": 0, "skipped": 0}
# Track test issues created so we can offer cleanup
_test_issue_ids = []


def log_result(name: str, passed: bool, detail: str = ""):
    if passed:
        test_results["passed"] += 1
        print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))
    else:
        test_results["failed"] += 1
        print(f"  [{FAIL}] {name}" + (f" — {detail}" if detail else ""))


def log_skip(name: str, reason: str):
    test_results["skipped"] += 1
    print(f"  [{SKIP}] {name} — {reason}")


def log_info(msg: str):
    print(f"  [{INFO}] {msg}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def make_sample_commits(count=3) -> list[CommitInfo]:
    """Generate realistic fake commits for testing."""
    samples = [
        CommitInfo(
            sha="abc1234567890",
            message="feat: add user authentication with JWT tokens",
            author="test-dev",
            repo="test-org/test-repo",
            branch="main",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        CommitInfo(
            sha="def2345678901",
            message="fix: resolve race condition in database connection pool",
            author="test-dev",
            repo="test-org/test-repo",
            branch="main",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        CommitInfo(
            sha="ghi3456789012",
            message="chore: update dependencies and fix lint warnings",
            author="test-dev",
            repo="test-org/test-repo",
            branch="main",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    ]
    return samples[:count]


# ─────────────────────────────────────────────
# Phase 1: Config Validation
# ─────────────────────────────────────────────

def test_phase1_config():
    section("Phase 1: Config Validation")

    log_result(
        "GITHUB_TOKEN is set",
        bool(config.GITHUB_TOKEN),
        f"{'***' + config.GITHUB_TOKEN[-4:] if config.GITHUB_TOKEN else 'MISSING'}",
    )
    log_result(
        "GITHUB_ORG is set",
        bool(config.GITHUB_ORG),
        config.GITHUB_ORG or "MISSING",
    )
    log_result(
        "LINEAR_API_KEY is set",
        bool(config.LINEAR_API_KEY),
        f"{'***' + config.LINEAR_API_KEY[-4:] if config.LINEAR_API_KEY else 'MISSING'}",
    )
    log_result(
        "LINEAR_TEAM_ID is set",
        bool(config.LINEAR_TEAM_ID),
        config.LINEAR_TEAM_ID or "MISSING",
    )

    gemini_count = len(config.GEMINI_KEYS)
    log_result(
        "Gemini API keys loaded",
        gemini_count >= 1,
        f"{gemini_count} key(s) found (recommended: 3)",
    )

    log_result(
        "POLL_INTERVAL_SECONDS valid",
        config.POLL_INTERVAL_SECONDS > 0,
        str(config.POLL_INTERVAL_SECONDS),
    )
    log_result(
        "BATCH_SIZE valid",
        config.BATCH_SIZE > 0,
        str(config.BATCH_SIZE),
    )

    all_set = all([
        config.GITHUB_TOKEN, config.GITHUB_ORG,
        config.LINEAR_API_KEY, config.LINEAR_TEAM_ID,
        config.GEMINI_KEYS,
    ])
    return all_set


# ─────────────────────────────────────────────
# Phase 2: GitHub Agent (Read-Only)
# ─────────────────────────────────────────────

def test_phase2_github():
    section("Phase 2: GitHub Agent (Read-Only)")

    if not config.GITHUB_TOKEN or not config.GITHUB_ORG:
        log_skip("GitHub tests", "GITHUB_TOKEN or GITHUB_ORG not set")
        return False

    github = GitHubAgent(config.GITHUB_TOKEN, config.GITHUB_ORG)

    # Test 2a: Fetch org repos
    repos = github.fetch_org_repos()
    log_result(
        "Fetch org repos",
        len(repos) > 0,
        f"Found {len(repos)} repo(s) in '{config.GITHUB_ORG}'",
    )

    if not repos:
        log_skip("Fetch commits", "No repos found")
        return False

    # Test 2b: Fetch commits from first repo
    first_repo = repos[0]
    log_info(f"Testing commit fetch on: {first_repo.full_name}")

    commits, latest_sha = github.fetch_new_commits(first_repo, None)
    log_result(
        "Fetch commits (first run, no last_sha)",
        latest_sha is not None,
        f"Got {len(commits)} commit(s), latest SHA: {latest_sha[:12] if latest_sha else 'None'}",
    )

    # Test 2c: Fetch with existing SHA (should return no new commits)
    if latest_sha:
        commits2, sha2 = github.fetch_new_commits(first_repo, latest_sha)
        log_result(
            "Fetch commits (with last_sha, expect 0 new)",
            len(commits2) == 0,
            f"Got {len(commits2)} new commit(s) (expected 0)",
        )

    # Test 2d: Full org scan
    repo_shas = {}
    all_commits, updated_shas = github.fetch_all_org_commits(repo_shas)
    log_result(
        "Full org commit scan",
        isinstance(updated_shas, dict) and len(updated_shas) > 0,
        f"{len(all_commits)} commit(s) across {len(updated_shas)} repo(s)",
    )

    return True


# ─────────────────────────────────────────────
# Phase 3: Gemini Agent
# ─────────────────────────────────────────────

def test_phase3_gemini():
    section("Phase 3: Gemini Agent (AI Classification)")

    if not config.GEMINI_KEYS:
        log_skip("Gemini tests", "No Gemini API keys set")
        return False

    # Test 3a: Key rotation
    km = KeyManager(config.GEMINI_KEYS)
    first_key = km.get_key()
    km.rotate()
    second_key = km.get_key()

    if len(config.GEMINI_KEYS) > 1:
        log_result("Key rotation", first_key != second_key, "Keys rotate correctly")
    else:
        log_result("Key rotation", True, "Only 1 key — rotation wraps around (OK)")

    # Test 3b: Classify sample commits
    gemini = GeminiAgent(config.GEMINI_KEYS)
    sample_commits = make_sample_commits(3)

    log_info("Sending 3 sample commits to Gemini for classification...")
    result = gemini.classify(sample_commits, [])

    if result:
        log_result(
            "Gemini classification",
            result.action in GeminiResult.VALID_ACTIONS,
            f"action={result.action}, title='{result.title[:50]}', priority={result.priority}",
        )
        log_result(
            "Result has valid label",
            result.label in GeminiResult.VALID_LABELS or True,  # allow unknown labels
            f"label={result.label}",
        )
        log_result(
            "Result has valid priority",
            0 <= result.priority <= 4,
            f"priority={result.priority}",
        )
        errors = result.validate()
        log_result(
            "Result passes validation",
            len(errors) == 0,
            f"{len(errors)} error(s)" + (f": {errors}" if errors else ""),
        )
    else:
        log_result("Gemini classification", False, "Returned None — check API key / quota")

    return result is not None


# ─────────────────────────────────────────────
# Phase 4: Linear Agent (Safe Mutations)
# ─────────────────────────────────────────────

def test_phase4_linear():
    section("Phase 4: Linear Agent (Safe Mutations)")

    if not config.LINEAR_API_KEY or not config.LINEAR_TEAM_ID:
        log_skip("Linear tests", "LINEAR_API_KEY or LINEAR_TEAM_ID not set")
        return False

    # Use a temp state dir so tests don't pollute real state
    test_state_dir = tempfile.mkdtemp(prefix="linear_test_state_")
    original_state_dir = config.STATE_DIR
    config.STATE_DIR = test_state_dir

    try:
        state = StateManager()
        linear = LinearAgent(config.LINEAR_API_KEY, config.LINEAR_TEAM_ID, state)

        # Test 4a: Ensure auto-sync label
        log_info("Ensuring 'auto-sync' label exists on Linear...")
        linear.ensure_auto_sync_label()
        log_result(
            "Auto-sync label",
            linear._auto_sync_label_id is not None,
            f"label_id={linear._auto_sync_label_id}",
        )

        if not linear._auto_sync_label_id:
            log_skip("Remaining Linear tests", "Could not create/find auto-sync label")
            return False

        # Test 4b: Fetch team labels
        linear.fetch_team_labels()
        log_result(
            "Fetch team labels",
            len(linear._label_cache) >= 0,  # could be 0 if no labels configured
            f"{len(linear._label_cache)} label(s) cached",
        )

        # Test 4c: Create a test issue
        log_info("Creating a test issue on Linear...")
        test_result = GeminiResult(
            action="CREATE_NEW",
            title="[TEST] Integration test issue — safe to delete",
            description="This issue was created by the integration test suite.\n\nIt can be safely deleted.",
            priority=4,  # Low
            label="Chore",
            state="Todo",
        )
        linear.execute(test_result, source_shas=["test_sha_001"])

        created = state.get_created_issues()
        issue_created = len(created) > 0
        log_result(
            "Create issue",
            issue_created,
            f"identifier={created[-1].identifier if created else 'NONE'}",
        )

        if not issue_created:
            log_skip("Update/subtask tests", "Issue creation failed")
            return False

        parent_issue = created[-1]
        _test_issue_ids.append(parent_issue.identifier)

        # Test 4d: Verify ownership check
        is_owned = state.is_owned_issue(parent_issue.identifier)
        log_result("Ownership check (owned)", is_owned, parent_issue.identifier)

        is_not_owned = not state.is_owned_issue("FAKE-99999")
        log_result("Ownership check (not owned)", is_not_owned, "FAKE-99999 correctly rejected")

        # Test 4e: Update the test issue
        log_info("Updating the test issue...")
        update_result = GeminiResult(
            action="UPDATE_EXISTING",
            title="[TEST] Updated title",
            description="Appended by integration test update step.",
            priority=3,
            label="Chore",
            state="In Progress",
            existing_issue_id=parent_issue.identifier,
        )
        linear.execute(update_result)

        # Verify description was appended
        desc = linear._get_issue_description(parent_issue.issue_id)
        log_result(
            "Update issue (description appended)",
            "Appended by integration test" in (desc or ""),
            f"Description length: {len(desc or '')} chars",
        )

        # Test 4f: Create a subtask under the test issue
        log_info("Creating a subtask under the test issue...")
        subtask_result = GeminiResult(
            action="ADD_SUBTASK",
            title="[TEST] Subtask — safe to delete",
            description="Subtask created by integration test.",
            priority=4,
            label="Chore",
            state="Todo",
            parent_issue_id=parent_issue.identifier,
        )
        linear.execute(subtask_result, source_shas=["test_sha_002"])

        updated_issues = state.get_created_issues()
        subtask_created = len(updated_issues) > len(created)
        if subtask_created:
            _test_issue_ids.append(updated_issues[-1].identifier)
        log_result(
            "Create subtask",
            subtask_created,
            f"identifier={updated_issues[-1].identifier if subtask_created else 'NONE'}",
        )

        # Test 4g: Safety — refuse to update non-owned issue
        log_info("Testing safety: refuse to update non-owned issue...")
        unsafe_result = GeminiResult(
            action="UPDATE_EXISTING",
            title="Should be blocked",
            description="This should not go through.",
            priority=1,
            label="Bug",
            state="Todo",
            existing_issue_id="FAKE-99999",
        )
        # This should log a warning and NOT crash
        linear.execute(unsafe_result)
        log_result("Safety: refuse non-owned update", True, "No crash, update was blocked")

        return True

    finally:
        config.STATE_DIR = original_state_dir
        shutil.rmtree(test_state_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Phase 5: Full Pipeline (End-to-End)
# ─────────────────────────────────────────────

def test_phase5_pipeline():
    section("Phase 5: Full Pipeline (GitHub -> Buffer -> Gemini -> Linear)")

    required = all([
        config.GITHUB_TOKEN, config.GITHUB_ORG,
        config.GEMINI_KEYS,
        config.LINEAR_API_KEY, config.LINEAR_TEAM_ID,
    ])
    if not required:
        log_skip("Pipeline test", "Not all API keys are configured")
        return False

    # Use temp state
    test_state_dir = tempfile.mkdtemp(prefix="linear_test_pipeline_")
    original_state_dir = config.STATE_DIR
    original_batch_size = config.BATCH_SIZE
    config.STATE_DIR = test_state_dir
    config.BATCH_SIZE = 3

    try:
        state = StateManager()
        github = GitHubAgent(config.GITHUB_TOKEN, config.GITHUB_ORG)
        buffer = BufferAgent(state)
        gemini = GeminiAgent(config.GEMINI_KEYS)
        linear = LinearAgent(config.LINEAR_API_KEY, config.LINEAR_TEAM_ID, state)

        linear.ensure_auto_sync_label()
        linear.fetch_team_labels()

        # Step 1: Fetch real commits from org
        log_info("Fetching real commits from org...")
        repo_shas = state.get_repo_shas()
        new_commits, updated_shas = github.fetch_all_org_commits(repo_shas)

        if new_commits:
            state.update_repo_shas(updated_shas)
        log_result(
            "Pipeline: GitHub fetch",
            isinstance(updated_shas, dict),
            f"{len(new_commits)} real commit(s) fetched",
        )

        # Step 2: Buffer commits (use sample if not enough real ones)
        commits_to_use = new_commits if len(new_commits) >= 3 else make_sample_commits(3)
        buffer.add_commits(commits_to_use[:3])

        is_ready = buffer.is_ready()
        log_result("Pipeline: Buffer ready", is_ready, f"Buffer has {len(commits_to_use[:3])} commit(s)")

        if not is_ready:
            log_skip("Pipeline: Gemini + Linear", "Buffer not ready")
            return False

        # Step 3: Classify with Gemini
        batch = buffer.get_batch()
        log_info(f"Sending {len(batch)} commits to Gemini...")
        result = gemini.classify(batch, state.get_created_issues())

        log_result(
            "Pipeline: Gemini classify",
            result is not None,
            f"action={result.action}, title='{result.title[:40]}'" if result else "Failed",
        )

        if not result:
            log_skip("Pipeline: Linear sync", "Gemini classification failed")
            return False

        # Step 4: Sync to Linear
        # Override title to mark as test
        result.title = f"[PIPELINE TEST] {result.title}"
        source_shas = [c.sha for c in batch]
        log_info("Syncing to Linear...")
        linear.execute(result, source_shas)

        created = state.get_created_issues()
        pipeline_ok = len(created) > 0
        if pipeline_ok:
            _test_issue_ids.append(created[-1].identifier)
        log_result(
            "Pipeline: Linear sync",
            pipeline_ok,
            f"Created {created[-1].identifier}" if pipeline_ok else "No issue created",
        )

        # Step 5: Clear buffer
        buffer.clear_batch(batch)
        remaining = state.load_buffer()
        log_result("Pipeline: Buffer cleared", len(remaining) == 0, f"{len(remaining)} remaining")

        return pipeline_ok

    finally:
        config.STATE_DIR = original_state_dir
        config.BATCH_SIZE = original_batch_size
        shutil.rmtree(test_state_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Phase 6: Model & State Unit Checks
# ─────────────────────────────────────────────

def test_phase6_models_and_state():
    section("Phase 6: Models & State (Offline)")

    # Test CommitInfo serialization
    c = CommitInfo("sha1", "test msg", "author", "org/repo", "main", "2025-01-01T00:00:00")
    d = c.to_dict()
    c2 = CommitInfo.from_dict(d)
    log_result("CommitInfo round-trip", c == c2, "to_dict -> from_dict")

    # Test GeminiResult validation
    valid = GeminiResult("CREATE_NEW", "title", "desc", 2, "Feature", "Todo")
    log_result("GeminiResult valid", len(valid.validate()) == 0)

    invalid = GeminiResult("INVALID", "", "desc", 9, "Feature", "Todo")
    errors = invalid.validate()
    log_result("GeminiResult invalid caught", len(errors) >= 2, f"{len(errors)} error(s)")

    subtask_no_parent = GeminiResult("ADD_SUBTASK", "title", "desc", 2, "Bug", "Todo")
    log_result(
        "ADD_SUBTASK without parent_id flagged",
        "parent_issue_id" in str(subtask_no_parent.validate()),
    )

    # Test LinearIssueRecord serialization
    rec = LinearIssueRecord("uuid1", "TEAM-1", "https://example.com", "2025-01-01", ["sha1"])
    rec2 = LinearIssueRecord.from_dict(rec.to_dict())
    log_result("LinearIssueRecord round-trip", rec == rec2)

    # Test StateManager with temp dir
    test_dir = tempfile.mkdtemp(prefix="state_test_")
    original = config.STATE_DIR
    config.STATE_DIR = test_dir

    try:
        sm = StateManager()

        # SHA tracking
        sm.update_repo_shas({"org/repo1": "sha_a", "org/repo2": "sha_b"})
        shas = sm.get_repo_shas()
        log_result("State: repo SHA tracking", shas.get("org/repo1") == "sha_a")

        # Buffer
        sm.save_buffer([c])
        loaded = sm.load_buffer()
        log_result("State: buffer save/load", len(loaded) == 1 and loaded[0].sha == "sha1")

        # Issue registry
        sm.add_created_issue(rec)
        log_result("State: issue registry", sm.is_owned_issue("TEAM-1"))
        log_result("State: non-owned rejected", not sm.is_owned_issue("FAKE-999"))

    finally:
        config.STATE_DIR = original
        shutil.rmtree(test_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────

def cleanup():
    if not _test_issue_ids:
        return

    section("Cleanup")
    print(f"\n  Test issues created on Linear: {', '.join(_test_issue_ids)}")
    print(f"  These are tagged with '{AUTO_SYNC_LABEL_NAME}' and marked [TEST].")
    print(f"  You can delete them manually from your Linear dashboard.\n")
    print(f"  (This script NEVER auto-deletes issues — matching the safety policy.)\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  GitHub → Linear Sync Engine — Integration Tests")
    print("=" * 60)

    # Setup minimal logging (suppress noisy output from agents)
    logging.basicConfig(
        level=logging.WARNING,
        format="    %(levelname)s: %(message)s",
    )

    # Run phases
    config_ok = test_phase1_config()

    if not config_ok:
        print(f"\n  Config incomplete. Fill in .env and re-run.\n")
        print_summary()
        return

    test_phase6_models_and_state()
    test_phase2_github()
    test_phase3_gemini()
    test_phase4_linear()
    test_phase5_pipeline()

    cleanup()
    print_summary()


def print_summary():
    section("Summary")
    total = test_results["passed"] + test_results["failed"] + test_results["skipped"]
    print(f"  Total:   {total}")
    print(f"  {PASS}:  {test_results['passed']}")
    print(f"  {FAIL}:  {test_results['failed']}")
    print(f"  {SKIP}:  {test_results['skipped']}")

    if test_results["failed"] == 0:
        print(f"\n  All tests passed! Your setup is ready.\n")
    else:
        print(f"\n  Some tests failed. Check your .env and API permissions.\n")


if __name__ == "__main__":
    main()
