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
from state import StateManager, CacheManager
from agents import (
    GitHubAgent, BufferAgent, GeminiAgent, LinearAgent,
    MappingAgent, ImprovementAgent, GuardAgent,
)

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
    cache = CacheManager()
    github = GitHubAgent(config.GITHUB_TOKEN, config.GITHUB_ORG)
    buffer = BufferAgent(state)
    gemini = GeminiAgent(config.GEMINI_KEYS)
    linear = LinearAgent(config.LINEAR_API_KEY, config.LINEAR_TEAM_ID, state)
    mapping = MappingAgent(config.GEMINI_KEYS, cache)
    improvement = ImprovementAgent(config.GEMINI_KEYS, cache)
    linear.set_mapping_agent(mapping)
    linear.set_github_agent(github)
    gemini.set_improvement_agent(improvement)
    guard = GuardAgent(state)

    # 3. One-time setup — resolve team key to UUID
    linear.resolve_team()
    if not linear._team_uuid:
        logger.error("Could not resolve Linear team. Check LINEAR_TEAM_ID in .env")
        return
    # Each setup step is wrapped so a single failure doesn't block startup
    for setup_name, setup_fn in [
        ("ensure_auto_sync_label", linear.ensure_auto_sync_label),
        ("fetch_team_labels", linear.fetch_team_labels),
        ("fetch_workflow_states", linear.fetch_workflow_states),
        ("fetch_projects", linear.fetch_projects),
        ("fetch_cycles", linear.fetch_cycles),
        ("fetch_team_members", linear.fetch_team_members),
    ]:
        try:
            setup_fn()
        except Exception as e:
            logger.error(f"SETUP_ERROR | {setup_name} failed: {e}", exc_info=True)

    ws = linear.get_workspace_context()
    logger.info(
        f"Workspace context ready: {len(ws.get('members', []))} members, "
        f"{len(ws.get('workflow_states', []))} states, "
        f"{len(ws.get('labels', []))} labels, "
        f"{len(ws.get('projects', []))} projects"
    )

    # 4. Backfill: update previously created issues missing assignee/state/cycle
    try:
        linear.backfill_created_issues()
    except Exception as e:
        logger.error(f"BACKFILL_STARTUP_ERROR | {e}", exc_info=True)

    # 5. Main loop
    _last_workspace_refresh = time.time()
    while not _shutdown:
        try:
            # --- Step A: Fetch new commits from GitHub ---
            new_commits = []
            try:
                repo_shas = state.get_repo_shas()
                new_commits, updated_shas = github.fetch_all_org_commits(repo_shas)
                if updated_shas:
                    state.update_repo_shas(updated_shas)
            except Exception as e:
                logger.error(f"FETCH_COMMITS_ERROR | {e}", exc_info=True)

            # --- Step B: Buffer new commits ---
            try:
                if new_commits:
                    buffer.add_commits(new_commits)
                    logger.info(f"BUFFER_STATUS | added={len(new_commits)} | pending={buffer.pending_count()}")
            except Exception as e:
                logger.error(f"BUFFER_ERROR | {e}", exc_info=True)

            # --- Step C: Process ready batches ---
            while buffer.is_ready() and not _shutdown:
                batch = None
                try:
                    batch = buffer.get_batch()
                    all_created_issues = state.get_created_issues()

                    # Per-user correlation
                    primary_author = batch[0].author if batch else None
                    user_recent_issues = None
                    if primary_author:
                        try:
                            user_recent_issues = linear.fetch_user_recent_issues(primary_author)
                        except Exception as e:
                            logger.warning(f"USER_ISSUES_ERROR | author={primary_author} | {e}")

                    workspace_context = linear.get_workspace_context()
                    # Gemini sees ALL team tasks (labeled by owner) for full context;
                    # ownership enforcement happens in LinearAgent.execute()
                    result = gemini.classify(
                        batch, all_created_issues, user_recent_issues,
                        workspace_context, primary_author
                    )

                    if result:
                        source_shas = [c.sha for c in batch]

                        # Guard check — block spam/duplicates before Linear execution
                        # Guard uses ALL issues (SHA dedup must check across all users)
                        verdict = guard.evaluate(
                            result, source_shas, all_created_issues,
                            user_recent_issues, primary_author
                        )
                        if not verdict:
                            logger.info(
                                f"GUARD_BLOCKED | action={result.action} | "
                                f"check={verdict.check_name} | reason={verdict.reason}"
                            )
                            continue

                        try:
                            linear.execute(result, source_shas, primary_author)
                        except Exception as e:
                            logger.error(f"LINEAR_EXECUTE_ERROR | {result.action} | {result.title} | {e}", exc_info=True)

                        # Track classification for self-improvement
                        try:
                            improvement.record_classification(
                                task_identifier=result.title[:30],
                                gemini_result={
                                    "action": result.action,
                                    "priority": result.priority,
                                    "label": result.label,
                                    "state": result.state,
                                    "assignee": result.assignee,
                                    "project": result.project,
                                },
                            )
                        except Exception as e:
                            logger.warning(f"IMPROVEMENT_TRACK_ERROR | {e}")

                except Exception as e:
                    logger.error(f"BATCH_PROCESS_ERROR | {e}", exc_info=True)
                finally:
                    # Always clear the batch to prevent infinite retry loops
                    if batch:
                        try:
                            buffer.clear_batch(batch)
                        except Exception as e:
                            logger.error(f"BATCH_CLEAR_ERROR | {e}")

                # Cooldown between Gemini calls to respect rate limits
                if buffer.is_ready():
                    logger.debug("Cooling down 10s before next batch...")
                    time.sleep(10)

            # --- Step D: Periodic self-improvement ---
            try:
                if improvement.should_attempt_improvement():
                    from agents.gemini_agent import SYSTEM_PROMPT
                    improvement.generate_improvement(SYSTEM_PROMPT)
                    gemini.set_improvement_agent(improvement)
            except Exception as e:
                logger.warning(f"IMPROVEMENT_ERROR | {e}")

            # --- Step E: Periodic LLM-driven cache cleanup ---
            try:
                if cache.should_run_llm_cleanup():
                    entries = cache.get_entries_summary()
                    if entries:
                        keys_to_remove = gemini.evaluate_cache_cleanup(entries)
                        if keys_to_remove:
                            cache.remove_keys(keys_to_remove)
                    cache.mark_llm_cleanup_done()
            except Exception as e:
                logger.warning(f"CACHE_CLEANUP_ERROR | {e}")

            # --- Step F: Periodic workspace data refresh ---
            try:
                elapsed = time.time() - _last_workspace_refresh
                if elapsed >= config.WORKSPACE_REFRESH_SECONDS:
                    logger.info("WORKSPACE_REFRESH | Refreshing projects, states, labels, cycles, members...")
                    # Invalidate cached workspace data so fetch functions re-query the API
                    for cache_key in ["team_members", "team_members_detail"]:
                        cache.invalidate(cache_key)
                    for name, fn in [
                        ("fetch_team_labels", linear.fetch_team_labels),
                        ("fetch_workflow_states", linear.fetch_workflow_states),
                        ("fetch_projects", linear.fetch_projects),
                        ("fetch_cycles", linear.fetch_cycles),
                        ("fetch_team_members", linear.fetch_team_members),
                    ]:
                        try:
                            fn()
                        except Exception as e:
                            logger.warning(f"WORKSPACE_REFRESH_ERROR | {name}: {e}")
                    ws = linear.get_workspace_context()
                    logger.info(
                        f"WORKSPACE_REFRESH_DONE | {len(ws.get('members', []))} members, "
                        f"{len(ws.get('workflow_states', []))} states, "
                        f"{len(ws.get('labels', []))} labels, "
                        f"{len(ws.get('projects', []))} projects"
                    )
                    _last_workspace_refresh = time.time()
            except Exception as e:
                logger.warning(f"WORKSPACE_REFRESH_ERROR | {e}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"MAIN_LOOP_ERROR | {e}", exc_info=True)

        # Sleep in small increments so shutdown is responsive
        for _ in range(config.POLL_INTERVAL_SECONDS):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("Sync engine stopped.")


if __name__ == "__main__":
    main()
