"""
GitHub to Linear Sync Engine — Orchestrator
Coordinates all agents in a continuous polling loop.

Usage:
    python main.py
"""

import signal
import time
import logging

import config
from state import StateManager
from agents import GitHubAgent, BufferAgent, GeminiAgent, LinearAgent

logger = logging.getLogger("orchestrator")

# --- Graceful shutdown ---
_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown  # noqa: PLW0603
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main():
    # 1. Config & logging
    config.validate()
    config.setup_logging()
    logger.info("=" * 60)
    logger.info("GitHub to Linear Sync Engine starting")
    logger.info(f"  Org:           {config.GITHUB_ORG}")
    logger.info(f"  Team ID:       {config.LINEAR_TEAM_ID}")
    logger.info(f"  Poll interval: {config.POLL_INTERVAL_SECONDS}s")
    logger.info(f"  Batch size:    {config.BATCH_SIZE}")
    logger.info(f"  Batch timeout: {config.BATCH_TIMEOUT_SECONDS}s")
    logger.info(f"  Dry run:       {config.DRY_RUN}")
    logger.info(f"  Gemini keys:   {len(config.GEMINI_KEYS)}")
    logger.info("=" * 60)

    # 2. Init agents
    state = StateManager()
    github = GitHubAgent(config.GITHUB_TOKEN, config.GITHUB_ORG)
    buffer = BufferAgent(state)
    gemini = GeminiAgent(config.GEMINI_KEYS)
    linear = LinearAgent(config.LINEAR_API_KEY, config.LINEAR_TEAM_ID, state)

    # 3. One-time setup — resolve team key to UUID
    linear.resolve_team()
    if not linear._team_uuid:
        logger.error("Could not resolve Linear team. Check LINEAR_TEAM_ID in .env")
        return
    linear.ensure_auto_sync_label()
    linear.fetch_team_labels()
    linear.fetch_team_members()

    # 4. Main loop
    while not _shutdown:
        try:
            repo_shas = state.get_repo_shas()

            # Fetch new commits from all org repos
            new_commits, updated_shas = github.fetch_all_org_commits(repo_shas)

            # Always save SHA positions (even on first run with 0 new commits)
            if updated_shas:
                state.update_repo_shas(updated_shas)

            # Buffer new commits
            if new_commits:
                buffer.add_commits(new_commits)

            # Process all ready batches (one at a time with cooldown)
            while buffer.is_ready() and not _shutdown:
                batch = buffer.get_batch()
                created_issues = state.get_created_issues()

                # Per-user correlation: fetch the commit author's recent Linear tasks
                primary_author = batch[0].author if batch else None
                user_recent_issues = None
                if primary_author:
                    user_recent_issues = linear.fetch_user_recent_issues(primary_author)

                result = gemini.classify(batch, created_issues, user_recent_issues)

                if result:
                    source_shas = [c.sha for c in batch]
                    linear.execute(result, source_shas)

                buffer.clear_batch(batch)

                # Cooldown between Gemini calls to respect rate limits
                if buffer.is_ready():
                    logger.debug("Cooling down 10s before next batch...")
                    time.sleep(10)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)

        # Sleep in small increments so shutdown is responsive
        for _ in range(config.POLL_INTERVAL_SECONDS):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("Sync engine stopped.")


if __name__ == "__main__":
    main()
