"""
Linear Agent — SAFE
Creates and updates issues in Linear via GraphQL.
Safety rules:
  1. Every created issue gets the 'auto-sync' label.
  2. Only updates issues that exist in the local created_issues registry
     AND have the 'auto-sync' label on Linear.
  3. NEVER deletes issues.
  4. All mutations are single-issue (no batch operations).
"""

import logging
from datetime import datetime, timezone
import requests
from models import GeminiResult, LinearIssueRecord
from state import StateManager, CacheManager, USER_ISSUES_TTL, TEAM_MEMBERS_TTL
import config

logger = logging.getLogger("agent.linear")

AUTO_SYNC_LABEL_NAME = "auto-sync"
AUTO_SYNC_LABEL_COLOR = "#6B7280"


class LinearAgent:
    """Safe Linear integration agent. Creates/updates, never deletes."""

    def __init__(self, api_key: str, team_id: str, state: StateManager):
        self._api_key = api_key
        self._team_id = team_id          # Could be key like "LAT" or UUID
        self._team_uuid: str | None = None  # Resolved UUID
        self._state = state
        self._auto_sync_label_id: str | None = None
        self._label_cache: dict[str, str] = {}  # name -> id
        self._user_cache: dict[str, str] = {}  # lowercase display name -> linear user id
        self._cache = CacheManager()

    # --- Setup (run once on startup) ---

    def resolve_team(self):
        """Resolve team key/identifier to UUID. Must be called before other operations."""
        # Try finding the team by key first (e.g. "LAT"), then by id
        query = """
        query {
          teams {
            nodes { id key name }
          }
        }
        """
        data = self._gql(query)
        nodes = self._safe_get(data, "data", "teams", "nodes") or []

        for team in nodes:
            if team["id"] == self._team_id or team["key"] == self._team_id:
                self._team_uuid = team["id"]
                logger.info(f"Resolved team '{self._team_id}' → {team['name']} (UUID: {self._team_uuid})")
                return

        logger.error(
            f"Could not resolve team '{self._team_id}'. "
            f"Available teams: {[t['key'] for t in nodes]}"
        )

    def ensure_auto_sync_label(self):
        """Create the 'auto-sync' label if it doesn't exist."""
        if not self._team_uuid:
            logger.error("Team not resolved. Call resolve_team() first.")
            return

        query = """
        query {
          issueLabels(filter: { name: { eq: "%s" } }) {
            nodes { id name }
          }
        }
        """ % AUTO_SYNC_LABEL_NAME

        data = self._gql(query)
        nodes = self._safe_get(data, "data", "issueLabels", "nodes") or []

        if nodes:
            self._auto_sync_label_id = nodes[0]["id"]
            logger.info(f"Found existing '{AUTO_SYNC_LABEL_NAME}' label: {self._auto_sync_label_id}")
        else:
            mutation = """
            mutation {
              issueLabelCreate(input: {
                name: "%s",
                color: "%s",
                teamId: "%s"
              }) {
                success
                issueLabel { id name }
              }
            }
            """ % (AUTO_SYNC_LABEL_NAME, AUTO_SYNC_LABEL_COLOR, self._team_uuid)

            data = self._gql(mutation)
            result = self._safe_get(data, "data", "issueLabelCreate") or {}
            if result.get("success"):
                self._auto_sync_label_id = result["issueLabel"]["id"]
                logger.info(f"Created '{AUTO_SYNC_LABEL_NAME}' label: {self._auto_sync_label_id}")
            else:
                logger.error(f"Failed to create auto-sync label: {data}")

    def fetch_team_labels(self):
        """Cache all team labels (name -> id mapping)."""
        if not self._team_uuid:
            logger.error("Team not resolved. Call resolve_team() first.")
            return

        query = """
        query {
          issueLabels(filter: { team: { id: { eq: "%s" } } }) {
            nodes { id name }
          }
        }
        """ % self._team_uuid

        data = self._gql(query)
        nodes = self._safe_get(data, "data", "issueLabels", "nodes") or []
        self._label_cache = {n["name"]: n["id"] for n in nodes}
        logger.info(f"Cached {len(self._label_cache)} team labels")

    def fetch_team_members(self):
        """Cache team members: display name (lowercased) → Linear user ID.
        Uses JSON cache to avoid re-fetching within TEAM_MEMBERS_TTL."""
        if not self._team_uuid:
            logger.error("Team not resolved. Call resolve_team() first.")
            return

        # Check persistent cache first
        cached = self._cache.get("team_members", TEAM_MEMBERS_TTL)
        if cached:
            self._user_cache = cached
            logger.info(f"Loaded {len(self._user_cache)} team member lookup keys from cache")
            return

        query = """
        query {
          team(id: "%s") {
            members {
              nodes { id displayName email }
            }
          }
        }
        """ % self._team_uuid

        data = self._gql(query)
        nodes = self._safe_get(data, "data", "team", "members", "nodes") or []
        self._user_cache = {}
        for member in nodes:
            name = (member.get("displayName") or "").strip().lower()
            if name:
                self._user_cache[name] = member["id"]
            # Also index by first part of email (before @) for fallback matching
            email = (member.get("email") or "").strip().lower()
            if email and "@" in email:
                email_prefix = email.split("@")[0]
                if email_prefix not in self._user_cache:
                    self._user_cache[email_prefix] = member["id"]

        # Persist to JSON cache
        self._cache.set("team_members", self._user_cache)
        logger.info(f"Cached {len(nodes)} team member(s) ({len(self._user_cache)} lookup keys)")

    def _resolve_github_to_linear_user(self, github_username: str) -> str | None:
        """Try to map a GitHub username to a Linear user ID via display name matching."""
        username_lower = github_username.strip().lower()
        # Direct match on display name or email prefix
        if username_lower in self._user_cache:
            return self._user_cache[username_lower]
        # Substring match — e.g. GitHub "FazalulAbid" matching Linear "Fazalul Abid"
        for name, uid in self._user_cache.items():
            if username_lower in name or name in username_lower:
                return uid
        return None

    def fetch_user_recent_issues(self, github_username: str) -> list[dict]:
        """
        Fetch recent issues for a GitHub user (mapped to Linear user).
        Returns list of {identifier, title, state} dicts for Gemini context.
        Falls back to recent team issues if user can't be mapped.
        Results are cached per user for USER_ISSUES_TTL seconds.
        """
        cache_key = f"user_issues:{github_username.lower()}"

        # Check cache first
        cached = self._cache.get(cache_key, USER_ISSUES_TTL)
        if cached is not None:
            logger.debug(f"Cache hit for {github_username}'s recent issues ({len(cached)} issues)")
            return cached

        linear_user_id = self._resolve_github_to_linear_user(github_username)

        if linear_user_id:
            logger.debug(f"Mapped GitHub '{github_username}' → Linear user {linear_user_id}")
            query = """
            query {
              issues(
                filter: {
                  assignee: { id: { eq: "%s" } }
                  team: { id: { eq: "%s" } }
                }
                orderBy: updatedAt
                first: 15
              ) {
                nodes { identifier title state { name } }
              }
            }
            """ % (linear_user_id, self._team_uuid)
        else:
            logger.debug(
                f"Could not map GitHub '{github_username}' to Linear user. "
                f"Falling back to recent team issues."
            )
            query = """
            query {
              issues(
                filter: {
                  team: { id: { eq: "%s" } }
                }
                orderBy: updatedAt
                first: 10
              ) {
                nodes { identifier title state { name } }
              }
            }
            """ % self._team_uuid

        data = self._gql(query)
        nodes = self._safe_get(data, "data", "issues", "nodes") or []

        results = []
        for node in nodes:
            state_name = self._safe_get(node, "state", "name") or "Unknown"
            results.append({
                "identifier": node.get("identifier", ""),
                "title": node.get("title", ""),
                "state": state_name,
            })

        # Persist to cache
        self._cache.set(cache_key, results)

        logger.info(
            f"Fetched {len(results)} recent issue(s) for "
            f"{'user ' + github_username if linear_user_id else 'team (fallback)'}"
        )
        return results

    # --- Execute action based on Gemini result ---

    def execute(self, result: GeminiResult, source_shas: list[str] | None = None):
        """Route to the correct mutation based on action type."""
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] Would execute {result.action}: {result.title}")
            return

        if result.action == "CREATE_NEW":
            self._create_issue(result, source_shas or [])
        elif result.action == "ADD_SUBTASK":
            self._create_subtask(result, source_shas or [])
        elif result.action == "UPDATE_EXISTING":
            self._update_issue(result)
        else:
            logger.error(f"Unknown action: {result.action}. Skipping.")

    # --- Mutations ---

    def _create_issue(self, result: GeminiResult, source_shas: list[str]):
        """Create a new Linear issue with auto-sync label."""
        label_ids = self._resolve_label_ids(result.label)

        mutation = """
        mutation IssueCreate($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue { id identifier url }
          }
        }
        """
        variables = {
            "input": {
                "teamId": self._team_uuid,
                "title": result.title,
                "description": result.description,
                "priority": result.priority,
                "labelIds": label_ids,
            }
        }

        data = self._gql(mutation, variables)
        issue_data = self._safe_get(data, "data", "issueCreate") or {}

        if issue_data.get("success"):
            issue = issue_data["issue"]
            record = LinearIssueRecord(
                issue_id=issue["id"],
                identifier=issue["identifier"],
                url=issue["url"],
                created_at=datetime.now(timezone.utc).isoformat(),
                source_commits=source_shas,
            )
            self._state.add_created_issue(record)
            logger.info(f"Created issue {issue['identifier']}: {result.title}")
        else:
            logger.error(f"Failed to create issue: {data}")

    def _create_subtask(self, result: GeminiResult, source_shas: list[str]):
        """Create a sub-issue under a parent. Parent must be script-owned."""
        if not result.parent_issue_id:
            logger.error("ADD_SUBTASK but no parent_issue_id. Falling back to CREATE_NEW.")
            self._create_issue(result, source_shas)
            return

        # Safety: parent must be in our registry
        if not self._state.is_owned_issue(result.parent_issue_id):
            logger.warning(
                f"Parent {result.parent_issue_id} not in script registry. "
                f"Falling back to CREATE_NEW."
            )
            self._create_issue(result, source_shas)
            return

        parent_uuid = self._resolve_identifier_to_uuid(result.parent_issue_id)
        if not parent_uuid:
            logger.error(f"Could not resolve parent {result.parent_issue_id}. Falling back to CREATE_NEW.")
            self._create_issue(result, source_shas)
            return

        label_ids = self._resolve_label_ids(result.label)

        mutation = """
        mutation IssueCreate($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue { id identifier url }
          }
        }
        """
        variables = {
            "input": {
                "teamId": self._team_uuid,
                "title": result.title,
                "description": result.description,
                "priority": result.priority,
                "labelIds": label_ids,
                "parentId": parent_uuid,
            }
        }

        data = self._gql(mutation, variables)
        issue_data = self._safe_get(data, "data", "issueCreate") or {}

        if issue_data.get("success"):
            issue = issue_data["issue"]
            record = LinearIssueRecord(
                issue_id=issue["id"],
                identifier=issue["identifier"],
                url=issue["url"],
                created_at=datetime.now(timezone.utc).isoformat(),
                source_commits=source_shas,
            )
            self._state.add_created_issue(record)
            logger.info(
                f"Created subtask {issue['identifier']} under {result.parent_issue_id}: {result.title}"
            )
        else:
            logger.error(f"Failed to create subtask: {data}")

    def _update_issue(self, result: GeminiResult):
        """Update an existing issue. Must be script-owned (double-checked)."""
        if not result.existing_issue_id:
            logger.error("UPDATE_EXISTING but no existing_issue_id. Skipping.")
            return

        # Safety check 1: local registry
        if not self._state.is_owned_issue(result.existing_issue_id):
            logger.warning(
                f"Issue {result.existing_issue_id} not in script registry. "
                f"Refusing to update (safety)."
            )
            return

        # Safety check 2: verify auto-sync label on Linear
        if not self._verify_auto_sync_label(result.existing_issue_id):
            logger.warning(
                f"Issue {result.existing_issue_id} does not have '{AUTO_SYNC_LABEL_NAME}' label on Linear. "
                f"Refusing to update (safety)."
            )
            return

        issue_uuid = self._resolve_identifier_to_uuid(result.existing_issue_id)
        if not issue_uuid:
            logger.error(f"Could not resolve {result.existing_issue_id}. Skipping update.")
            return

        # Append to description rather than replacing
        existing_desc = self._get_issue_description(issue_uuid)
        separator = "\n\n---\n_Auto-sync update:_\n"
        new_desc = (existing_desc or "") + separator + result.description

        mutation = """
        mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) {
            success
            issue { id identifier }
          }
        }
        """
        variables = {
            "id": issue_uuid,
            "input": {
                "description": new_desc,
                "priority": result.priority,
            },
        }

        data = self._gql(mutation, variables)
        update_data = self._safe_get(data, "data", "issueUpdate") or {}

        if update_data.get("success"):
            logger.info(f"Updated issue {result.existing_issue_id}: {result.title}")
        else:
            logger.error(f"Failed to update issue: {data}")

    # --- Helpers ---

    def _resolve_label_ids(self, label_name: str) -> list[str]:
        """Resolve a label name to ID list, always including auto-sync."""
        ids = []
        if self._auto_sync_label_id:
            ids.append(self._auto_sync_label_id)
        if label_name in self._label_cache:
            label_id = self._label_cache[label_name]
            if label_id not in ids:
                ids.append(label_id)
        return ids

    def _resolve_identifier_to_uuid(self, identifier: str) -> str | None:
        """Resolve a human-readable identifier (e.g. TEAM-123) to a Linear UUID."""
        query = """
        query {
          issue(id: "%s") {
            id
          }
        }
        """ % identifier

        data = self._gql(query)
        issue = self._safe_get(data, "data", "issue")
        return issue["id"] if issue else None

    def _verify_auto_sync_label(self, identifier: str) -> bool:
        """Check if an issue on Linear has the auto-sync label."""
        query = """
        query {
          issue(id: "%s") {
            labels { nodes { name } }
          }
        }
        """ % identifier

        data = self._gql(query)
        issue = self._safe_get(data, "data", "issue")
        if not issue:
            return False
        nodes = self._safe_get(issue, "labels", "nodes") or []
        return any(n.get("name") == AUTO_SYNC_LABEL_NAME for n in nodes)

    def _get_issue_description(self, issue_uuid: str) -> str:
        """Fetch current description of an issue."""
        query = """
        query {
          issue(id: "%s") {
            description
          }
        }
        """ % issue_uuid

        data = self._gql(query)
        issue = self._safe_get(data, "data", "issue")
        return (issue.get("description") or "") if issue else ""

    @staticmethod
    def _safe_get(data: dict | None, *keys):
        """
        Safely traverse nested dicts where any value could be None.
        Returns None if any key is missing or any intermediate value is None.
        """
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL request against Linear API."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        resp = requests.post(
            config.LINEAR_API_URL,
            json=payload,
            headers={"Authorization": self._api_key},
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()

        # Log GraphQL-level errors
        if result.get("errors"):
            for err in result["errors"]:
                logger.warning(f"Linear GraphQL error: {err.get('message', err)}")

        return result
