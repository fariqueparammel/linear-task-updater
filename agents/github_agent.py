"""
GitHub Agent — READ-ONLY
Fetches commit history from all repositories in a GitHub organization.
Never writes, pushes, or modifies any repository.
"""

import logging
from github import Github, GithubException
from models import CommitInfo

logger = logging.getLogger("agent.github")

# Patterns to skip
MERGE_PREFIXES = ("Merge branch", "Merge pull request", "Merge remote-tracking")
BOT_MARKERS = ("[bot]", "dependabot", "renovate", "github-actions")


class GitHubAgent:
    """Read-only agent that fetches commits from all org repos."""

    def __init__(self, token: str, org_name: str):
        self._gh = Github(token)
        self._org_name = org_name
        logger.info(f"GitHubAgent initialized for org: {org_name}")

    def fetch_org_repos(self) -> list:
        """List all non-archived, non-fork repositories in the org."""
        try:
            org = self._gh.get_organization(self._org_name)
            repos = []
            for repo in org.get_repos(type="all"):
                if repo.archived or repo.fork:
                    continue
                repos.append(repo)
            logger.info(f"Found {len(repos)} active repos in {self._org_name}")
            return repos
        except GithubException as e:
            logger.error(f"Failed to fetch org repos: {e}")
            return []

    def fetch_new_commits(self, repo, last_sha: str | None) -> tuple[list[CommitInfo], str | None]:
        """
        Fetch new commits from a repo's default branch since last_sha.
        Returns (list_of_commits_oldest_first, latest_sha).
        """
        try:
            branch = repo.default_branch
            commits_page = repo.get_commits(sha=branch)

            new_commits = []
            latest_sha = None

            for commit in commits_page:
                sha = commit.sha

                # Set the latest SHA from the first (newest) commit
                if latest_sha is None:
                    latest_sha = sha

                # Stop when we reach the last processed commit
                if sha == last_sha:
                    break

                # If no last_sha, only take the very first commit (don't backfill)
                if last_sha is None:
                    new_commits.append(self._to_commit_info(commit, repo, branch))
                    break

                if self._should_skip(commit):
                    continue

                new_commits.append(self._to_commit_info(commit, repo, branch))

            # Return oldest-first order
            new_commits.reverse()

            if new_commits:
                logger.info(f"{repo.full_name}: {len(new_commits)} new commit(s)")
            return new_commits, latest_sha or last_sha

        except GithubException as e:
            logger.error(f"Failed to fetch commits for {repo.full_name}: {e}")
            return [], last_sha

    def fetch_all_org_commits(
        self, repo_shas: dict[str, str]
    ) -> tuple[list[CommitInfo], dict[str, str]]:
        """
        Iterate all org repos and fetch new commits for each.
        Returns (aggregated_commits, updated_sha_map).
        """
        repos = self.fetch_org_repos()
        all_commits = []
        updated_shas = {}

        for repo in repos:
            last_sha = repo_shas.get(repo.full_name)
            new_commits, latest_sha = self.fetch_new_commits(repo, last_sha)

            if latest_sha:
                updated_shas[repo.full_name] = latest_sha

            all_commits.extend(new_commits)

        if all_commits:
            logger.info(f"Total new commits across org: {len(all_commits)}")

        return all_commits, updated_shas

    def _to_commit_info(self, commit, repo, branch: str) -> CommitInfo:
        author_name = "unknown"
        if commit.author:
            author_name = commit.author.login
        elif commit.commit.author:
            author_name = commit.commit.author.name

        return CommitInfo(
            sha=commit.sha,
            message=commit.commit.message.strip(),
            author=author_name,
            repo=repo.full_name,
            branch=branch,
            timestamp=commit.commit.author.date.isoformat(),
        )

    def _should_skip(self, commit) -> bool:
        """Filter out merge commits, bot commits, and empty messages."""
        msg = commit.commit.message.strip()

        if not msg:
            return True

        if any(msg.startswith(prefix) for prefix in MERGE_PREFIXES):
            return True

        author = ""
        if commit.author:
            author = commit.author.login.lower()
        elif commit.commit.author:
            author = commit.commit.author.name.lower()

        if any(bot in author for bot in BOT_MARKERS):
            return True

        return False
