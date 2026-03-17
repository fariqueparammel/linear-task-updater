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

BOT_NAME = "CommitPilot"
BOT_SIGNATURE = f"\n\n---\n_Created by **{BOT_NAME}** — automated commit-to-task sync_"


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
        self._state_cache: dict[str, str] = {}  # state name -> state id
        self._project_cache: dict[str, str] = {}  # project name -> project id
        self._cycle_cache: dict[str, str] = {}  # cycle name -> cycle id
        self._active_cycle_id: str | None = None  # current active cycle UUID
        self._members_detail: list[dict] = []  # [{id, displayName, email}] for Gemini context
        self._cache = CacheManager()
        self._mapping_agent = None  # Set via set_mapping_agent()
        self._github_agent = None   # Set via set_github_agent()

    def set_mapping_agent(self, mapping_agent):
        """Inject the MappingAgent for AI-powered user resolution fallback."""
        self._mapping_agent = mapping_agent

    def set_github_agent(self, github_agent):
        """Inject the GitHubAgent for commit author lookups during backfill."""
        self._github_agent = github_agent

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

    def fetch_workflow_states(self):
        """Cache workflow states for the team (name → id mapping)."""
        if not self._team_uuid:
            return

        query = """
        query {
          workflowStates(filter: { team: { id: { eq: "%s" } } }) {
            nodes { id name type }
          }
        }
        """ % self._team_uuid

        data = self._gql(query)
        nodes = self._safe_get(data, "data", "workflowStates", "nodes") or []
        self._state_cache = {n["name"]: n["id"] for n in nodes}
        state_names = [f"{n['name']} ({n['type']})" for n in nodes]
        logger.info(f"Cached {len(self._state_cache)} workflow states: {', '.join(state_names)}")

    def fetch_projects(self):
        """Cache projects for the team (name → id mapping)."""
        if not self._team_uuid:
            return

        query = """
        query {
          projects(
            filter: { accessibleTeams: { id: { eq: "%s" } } }
            first: 50
          ) {
            nodes { id name state }
          }
        }
        """ % self._team_uuid

        data = self._gql(query)
        nodes = self._safe_get(data, "data", "projects", "nodes") or []
        # Only cache active projects (planned, started)
        active = [n for n in nodes if n.get("state") in ("planned", "started", None)]
        self._project_cache = {n["name"]: n["id"] for n in active}
        logger.info(f"Cached {len(self._project_cache)} active project(s)")

    def fetch_cycles(self):
        """Cache active/upcoming cycles for the team. Auto-assigns to current active cycle."""
        if not self._team_uuid:
            return

        # Query cycles through the team object (Linear API requires this path)
        query = """
        query {
          team(id: "%s") {
            cycles(first: 10) {
              nodes { id name number startsAt endsAt }
            }
          }
        }
        """ % self._team_uuid

        try:
            data = self._gql(query)
            nodes = self._safe_get(data, "data", "team", "cycles", "nodes") or []

            now = datetime.now(timezone.utc).isoformat()

            self._cycle_cache = {}
            for n in nodes:
                cycle_name = n.get("name") or f"Cycle {n.get('number', '?')}"
                self._cycle_cache[cycle_name] = n["id"]
                # Detect the currently active cycle
                starts = n.get("startsAt", "")
                ends = n.get("endsAt", "")
                if starts and ends and starts <= now <= ends:
                    self._active_cycle_id = n["id"]
                    logger.info(f"Active cycle detected: {cycle_name} ({n['id'][:8]}...)")

            logger.info(f"Cached {len(self._cycle_cache)} cycle(s), active={self._active_cycle_id is not None}")
        except Exception as e:
            logger.warning(f"Could not fetch cycles (non-critical): {e}")
            self._cycle_cache = {}

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
            self._members_detail = self._cache.get("team_members_detail", TEAM_MEMBERS_TTL) or []
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

        # Store full detail list for Gemini context
        self._members_detail = [
            {
                "id": m["id"],
                "displayName": m.get("displayName", ""),
                "email": m.get("email", ""),
            }
            for m in nodes
        ]

        # Persist to JSON cache
        self._cache.set("team_members", self._user_cache)
        self._cache.set("team_members_detail", self._members_detail)
        logger.info(f"Cached {len(nodes)} team member(s) ({len(self._user_cache)} lookup keys)")

    def fetch_project_assignees(self) -> list[dict]:
        """
        Fetch who works on which project by checking recent issue assignees.
        Returns list of {project, assignees: [displayName, ...]} for mapping context.
        """
        if not self._team_uuid or not self._project_cache:
            return []

        results = []
        for project_name, project_id in list(self._project_cache.items())[:10]:
            try:
                query = """
                query {
                  issues(
                    filter: {
                      team: { id: { eq: "%s" } }
                      project: { id: { eq: "%s" } }
                    }
                    first: 20
                    orderBy: updatedAt
                  ) {
                    nodes {
                      assignee { displayName }
                    }
                  }
                }
                """ % (self._team_uuid, project_id)

                data = self._gql(query)
                nodes = self._safe_get(data, "data", "issues", "nodes") or []

                assignees = set()
                for node in nodes:
                    assignee = self._safe_get(node, "assignee", "displayName")
                    if assignee:
                        assignees.add(assignee)

                if assignees:
                    results.append({
                        "project": project_name,
                        "assignees": sorted(assignees),
                    })
            except Exception as e:
                logger.debug(f"Could not fetch assignees for project '{project_name}': {e}")

        logger.info(f"Fetched project assignees for {len(results)} project(s)")
        return results

    def get_workspace_context(self) -> dict:
        """
        Return workspace metadata for Gemini prompt context.
        Includes team members, workflow states, labels, projects, and cycles.
        """
        return {
            "members": [
                {"displayName": m["displayName"], "email": m.get("email", "")}
                for m in self._members_detail
            ],
            "workflow_states": list(self._state_cache.keys()),
            "labels": list(self._label_cache.keys()),
            "projects": list(self._project_cache.keys()),
            "cycles": list(self._cycle_cache.keys()),
            "has_active_cycle": self._active_cycle_id is not None,
        }

    def resolve_assignee_id(self, assignee_name: str | None) -> str | None:
        """Resolve an assignee display name (from Gemini) to a Linear user UUID."""
        if not assignee_name:
            return None

        name_lower = assignee_name.strip().lower()

        # Direct match
        if name_lower in self._user_cache:
            logger.debug(f"ASSIGNEE_RESOLVE | '{assignee_name}' → direct match in user cache")
            return self._user_cache[name_lower]

        # Substring / fuzzy match
        for cached_name, uid in self._user_cache.items():
            if name_lower in cached_name or cached_name in name_lower:
                logger.debug(
                    f"ASSIGNEE_RESOLVE | '{assignee_name}' → substring match "
                    f"(cached key: '{cached_name}')"
                )
                return uid

        # Check against full member details (displayName exact match, case-insensitive)
        for m in self._members_detail:
            if m["displayName"].strip().lower() == name_lower:
                logger.debug(f"ASSIGNEE_RESOLVE | '{assignee_name}' → member detail match")
                return m["id"]

        logger.warning(
            f"ASSIGNEE_RESOLVE_FAILED | Could not resolve '{assignee_name}' to a Linear user. "
            f"Known keys: {list(self._user_cache.keys())}"
        )
        return None

    def resolve_state_id(self, state_name: str | None) -> str | None:
        """Resolve a workflow state name to its Linear UUID."""
        if not state_name:
            return None
        # Exact match
        if state_name in self._state_cache:
            return self._state_cache[state_name]
        # Case-insensitive match
        name_lower = state_name.strip().lower()
        for cached_name, sid in self._state_cache.items():
            if cached_name.lower() == name_lower:
                return sid
        return None

    def resolve_project_id(self, project_name: str | None) -> str | None:
        """Resolve a project name to its Linear UUID."""
        if not project_name:
            return None
        if project_name in self._project_cache:
            return self._project_cache[project_name]
        name_lower = project_name.strip().lower()
        for cached_name, pid in self._project_cache.items():
            if cached_name.lower() == name_lower:
                return pid
        return None

    def _resolve_github_to_linear_user(self, github_username: str) -> str | None:
        """
        Map a GitHub username to a Linear user ID.
        1. Direct name/email match
        2. Substring match
        3. Process of elimination — if all other members are mapped, the remaining one is this user
        4. MappingAgent AI-powered resolution (project context + elimination)
        """
        username_lower = github_username.strip().lower()
        # Direct match on display name or email prefix
        if username_lower in self._user_cache:
            return self._user_cache[username_lower]
        # Substring match — e.g. GitHub "FazalulAbid" matching Linear "Fazalul Abid"
        for name, uid in self._user_cache.items():
            if username_lower in name or name in username_lower:
                return uid

        # Process of elimination: gather already-mapped Linear user IDs, find unmapped members
        known_mappings = {}
        if self._mapping_agent:
            known_mappings = self._mapping_agent.get_known_mappings()
        mapped_user_ids = set()
        for _gh_user, linear_name in known_mappings.items():
            for m in self._members_detail:
                if m["displayName"].lower() == linear_name.lower():
                    mapped_user_ids.add(m["id"])
                    break
        unmapped_members = [
            m for m in self._members_detail if m["id"] not in mapped_user_ids
        ]
        if len(unmapped_members) == 1:
            match = unmapped_members[0]
            uid = match["id"]
            display_name = match["displayName"]
            logger.info(
                f"ASSIGNEE_ELIMINATION | '{github_username}' → '{display_name}' "
                f"(only unmapped member remaining out of {len(self._members_detail)})"
            )
            # Cache for future fast lookups
            self._user_cache[username_lower] = uid
            self._user_cache[display_name.lower()] = uid
            self._cache.set("team_members", self._user_cache)
            return uid
        elif unmapped_members:
            logger.info(
                f"ASSIGNEE_ELIMINATION | '{github_username}' has {len(unmapped_members)} "
                f"unmapped candidates: {[m['displayName'] for m in unmapped_members]}. "
                f"Cannot determine by elimination alone."
            )

        # Fallback: use MappingAgent with project context
        if self._mapping_agent and self._members_detail:
            logger.info(
                f"Direct mapping failed for '{github_username}'. "
                f"Invoking MappingAgent for AI-powered resolution..."
            )
            project_context = self.fetch_project_assignees()

            result = self._mapping_agent.resolve(
                github_username=github_username,
                members_detail=self._members_detail,
                project_context=project_context,
                known_mappings=known_mappings if known_mappings else None,
            )

            if result and result.get("linear_user_id"):
                # Cache the resolved mapping in _user_cache for future fast lookups
                uid = result["linear_user_id"]
                display_name = result["display_name"]
                self._user_cache[username_lower] = uid
                self._user_cache[display_name.lower()] = uid
                # Persist updated user cache
                self._cache.set("team_members", self._user_cache)
                logger.info(
                    f"MappingAgent resolved: '{github_username}' → "
                    f"'{display_name}' ({uid[:8]}...)"
                )
                return uid

        logger.warning(
            f"ASSIGNEE_UNRESOLVED | '{github_username}' could not be mapped to any "
            f"Linear member. Issue will be created without assignee."
        )
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

    # --- Backfill missing fields on previously created issues ---

    def backfill_created_issues(self):
        """
        Check ALL auto-sync labeled issues in Linear (not just local registry)
        and update any that are missing assignee, state, cycle, or project.
        Also registers any discovered issues not in the local state.
        """
        # Gather issues from both sources: local registry + Linear scan
        issues_to_check = self._gather_backfill_issues()
        if not issues_to_check:
            logger.info("BACKFILL | No auto-sync issues to check")
            return

        logger.info(f"BACKFILL_START | Checking {len(issues_to_check)} auto-sync issue(s)")
        updated_count = 0

        for issue_info in issues_to_check:
            try:
                issue_id = issue_info["id"]
                identifier = issue_info["identifier"]
                update_input = {}

                # Check missing assignee
                if not issue_info.get("assignee_id"):
                    assignee_id = self._resolve_backfill_assignee_from_info(issue_info)
                    if assignee_id:
                        update_input["assigneeId"] = assignee_id

                # Check missing cycle
                if not issue_info.get("cycle_id") and self._active_cycle_id:
                    update_input["cycleId"] = self._active_cycle_id

                # Check if state is default (Backlog)
                if issue_info.get("state_name") == "Backlog":
                    todo_id = self.resolve_state_id("Todo")
                    if todo_id:
                        update_input["stateId"] = todo_id

                # Check missing project — deduce from title/description
                if not issue_info.get("project_id") and self._project_cache:
                    project_id = self._deduce_project(
                        issue_info.get("title", ""),
                        issue_info.get("description", ""),
                    )
                    if project_id:
                        update_input["projectId"] = project_id

                if not update_input:
                    continue

                mutation = """
                mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
                  issueUpdate(id: $id, input: $input) {
                    success
                    issue { id identifier }
                  }
                }
                """
                variables = {"id": issue_id, "input": update_input}

                result = self._gql(mutation, variables)
                success = self._safe_get(result, "data", "issueUpdate", "success")

                if success:
                    fields = list(update_input.keys())
                    logger.info(f"BACKFILL_UPDATED | {identifier} | fields={fields}")
                    updated_count += 1
                else:
                    logger.warning(f"BACKFILL_FAILED | {identifier}: {result}")

            except Exception as e:
                logger.warning(f"BACKFILL_ERROR | {issue_info.get('identifier', '?')}: {e}")

        logger.info(f"BACKFILL_COMPLETE | Updated {updated_count}/{len(issues_to_check)} issue(s)")

    def _gather_backfill_issues(self) -> list[dict]:
        """
        Gather all auto-sync issues from both the local registry and Linear.
        Returns normalized list of dicts with issue details.
        Registers any Linear-discovered issues not in local state.
        """
        seen_ids = set()
        issues = []

        # Source 1: Local registry (has commit_author + source_commits)
        for record in self._state.get_created_issues():
            try:
                query = """
                query {
                  issue(id: "%s") {
                    id identifier title description
                    assignee { id displayName }
                    state { id name }
                    cycle { id }
                    project { id }
                  }
                }
                """ % record.issue_id

                data = self._gql(query)
                issue = self._safe_get(data, "data", "issue")
                if not issue:
                    continue

                seen_ids.add(issue["id"])
                issues.append({
                    "id": issue["id"],
                    "identifier": issue.get("identifier", record.identifier),
                    "title": issue.get("title", ""),
                    "description": issue.get("description", ""),
                    "assignee_id": self._safe_get(issue, "assignee", "id"),
                    "state_name": self._safe_get(issue, "state", "name"),
                    "cycle_id": self._safe_get(issue, "cycle", "id"),
                    "project_id": self._safe_get(issue, "project", "id"),
                    "commit_author": record.commit_author,
                    "source_commits": record.source_commits,
                })
            except Exception as e:
                logger.debug(f"BACKFILL | Could not fetch {record.identifier}: {e}")

        # Source 2: Scan Linear for auto-sync labeled issues not in local registry
        if self._auto_sync_label_id and self._team_uuid:
            try:
                query = """
                query {
                  issues(
                    filter: {
                      team: { id: { eq: "%s" } }
                      labels: { id: { eq: "%s" } }
                    }
                    first: 50
                    orderBy: createdAt
                  ) {
                    nodes {
                      id identifier title description
                      assignee { id displayName }
                      state { id name }
                      cycle { id }
                      project { id }
                    }
                  }
                }
                """ % (self._team_uuid, self._auto_sync_label_id)

                data = self._gql(query)
                nodes = self._safe_get(data, "data", "issues", "nodes") or []

                new_discoveries = 0
                for issue in nodes:
                    if issue["id"] in seen_ids:
                        continue

                    seen_ids.add(issue["id"])
                    issues.append({
                        "id": issue["id"],
                        "identifier": issue.get("identifier", "?"),
                        "title": issue.get("title", ""),
                        "description": issue.get("description", ""),
                        "assignee_id": self._safe_get(issue, "assignee", "id"),
                        "state_name": self._safe_get(issue, "state", "name"),
                        "cycle_id": self._safe_get(issue, "cycle", "id"),
                        "project_id": self._safe_get(issue, "project", "id"),
                        "commit_author": None,
                        "source_commits": [],
                    })

                    # Register in local state so future backfills are faster
                    record = LinearIssueRecord(
                        issue_id=issue["id"],
                        identifier=issue.get("identifier", "?"),
                        url=f"https://linear.app/issue/{issue.get('identifier', '')}",
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    self._state.add_created_issue(record)
                    new_discoveries += 1

                if new_discoveries:
                    logger.info(
                        f"BACKFILL_DISCOVERED | Found {new_discoveries} auto-sync issue(s) "
                        f"not in local registry — registered"
                    )
            except Exception as e:
                logger.warning(f"BACKFILL | Linear scan failed (non-critical): {e}")

        return issues

    def _resolve_backfill_assignee_from_info(self, issue_info: dict) -> str | None:
        """
        Resolve the assignee for a backfill issue.
        1. Use stored commit_author if available
        2. Look up source commit SHAs via GitHub API
        3. Try to extract author from the issue description (CommitPilot format)
        4. Map the GitHub username to a Linear member
        """
        github_username = issue_info.get("commit_author")

        # Try GitHub SHA lookup
        if not github_username and issue_info.get("source_commits") and self._github_agent:
            for sha in issue_info["source_commits"][:3]:
                try:
                    github_username = self._github_agent.fetch_commit_author(sha)
                    if github_username:
                        logger.debug(
                            f"BACKFILL | Found author '{github_username}' "
                            f"for {issue_info['identifier']} via SHA {sha[:8]}"
                        )
                        break
                except Exception:
                    pass

        # Try extracting author from description (CommitPilot descriptions contain commit info)
        if not github_username:
            description = issue_info.get("description", "")
            github_username = self._extract_author_from_description(description)

        if not github_username:
            logger.debug(
                f"BACKFILL | No author found for {issue_info['identifier']}, skipping assignee"
            )
            return None

        linear_user_id = self._resolve_github_to_linear_user(github_username)
        if linear_user_id:
            logger.info(
                f"BACKFILL_ASSIGNEE | {issue_info['identifier']} | "
                f"github={github_username} → linear_user={linear_user_id[:8]}..."
            )
        return linear_user_id

    @staticmethod
    def _extract_author_from_description(description: str) -> str | None:
        """
        Try to extract a GitHub username from a CommitPilot-generated description.
        Looks for patterns like 'by username' or 'author: username' in the text.
        """
        if not description:
            return None
        import re
        # Match patterns like "by FazalulAbid" or "Commit 1 (by username,"
        match = re.search(r'\(by\s+(\w[\w-]*)', description)
        if match:
            return match.group(1)
        # Match "author: username" or "Author: username"
        match = re.search(r'[Aa]uthor[:\s]+(\w[\w-]*)', description)
        if match:
            return match.group(1)
        return None

    def _deduce_project(self, title: str, description: str) -> str | None:
        """
        Try to match an issue to a project based on title/description keywords.
        Uses simple keyword matching against project names.
        """
        if not self._project_cache:
            return None

        text = (title + " " + (description or "")).lower()

        best_match = None
        best_score = 0

        for project_name, project_id in self._project_cache.items():
            # Split project name into keywords
            keywords = project_name.lower().split()
            # Count how many project name words appear in the issue text
            score = sum(1 for kw in keywords if len(kw) > 2 and kw in text)
            # Also check if the full project name appears
            if project_name.lower() in text:
                score += len(keywords) * 2  # Strong boost for full name match

            if score > best_score:
                best_score = score
                best_match = (project_name, project_id)

        # Require at least 1 keyword match (for single-word names) or 2+ for multi-word
        if best_match:
            project_name, project_id = best_match
            min_score = 1 if len(project_name.split()) == 1 else 2
            if best_score >= min_score:
                logger.info(f"BACKFILL_PROJECT | Deduced project '{project_name}' (score={best_score})")
                return project_id

        return None

    # --- Execute action based on Gemini result ---

    def execute(self, result: GeminiResult, source_shas: list[str] | None = None, primary_author: str | None = None):
        """Route to the correct mutation based on action type."""
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] Would execute {result.action}: {result.title}")
            return

        # Per-user ownership: for UPDATE/SUBTASK, verify target belongs to same author
        if primary_author and result.action in ("UPDATE_EXISTING", "ADD_SUBTASK"):
            target_id = result.existing_issue_id if result.action == "UPDATE_EXISTING" else result.parent_issue_id
            if target_id:
                target_record = self._state.get_issue_by_identifier(target_id)
                if target_record and target_record.commit_author and target_record.commit_author != primary_author:
                    logger.warning(
                        f"OWNERSHIP_MISMATCH | {result.action} targets {target_id} "
                        f"(owner={target_record.commit_author}) but commit author is "
                        f"{primary_author}. Falling back to CREATE_NEW."
                    )
                    result.action = "CREATE_NEW"
                    result.existing_issue_id = None
                    result.parent_issue_id = None

        if result.action == "CREATE_NEW":
            self._create_issue(result, source_shas or [], primary_author)
        elif result.action == "ADD_SUBTASK":
            self._create_subtask(result, source_shas or [], primary_author)
        elif result.action == "UPDATE_EXISTING":
            self._update_issue(result)
        else:
            logger.error(f"Unknown action: {result.action}. Skipping.")

    # --- Mutations ---

    def _create_issue(self, result: GeminiResult, source_shas: list[str], primary_author: str | None = None):
        """Create a new Linear issue with auto-sync label."""
        label_ids = self._resolve_label_ids(result.label)
        assignee_id = self.resolve_assignee_id(result.assignee)

        # Fallback: if Gemini couldn't match the assignee, resolve from GitHub username
        if not assignee_id and primary_author:
            logger.info(
                f"ASSIGNEE_FALLBACK | Gemini assignee '{result.assignee}' not resolved. "
                f"Trying GitHub username '{primary_author}' → Linear user mapping..."
            )
            assignee_id = self._resolve_github_to_linear_user(primary_author)
            if assignee_id:
                logger.info(f"ASSIGNEE_FALLBACK_OK | '{primary_author}' resolved via GitHub→Linear mapping")

        state_id = self.resolve_state_id(result.state)
        project_id = self.resolve_project_id(result.project)

        mutation = """
        mutation IssueCreate($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue { id identifier url }
          }
        }
        """
        description = result.description + BOT_SIGNATURE

        input_vars = {
            "teamId": self._team_uuid,
            "title": result.title,
            "description": description,
            "priority": result.priority,
            "labelIds": label_ids,
        }
        if assignee_id:
            input_vars["assigneeId"] = assignee_id
        if state_id:
            input_vars["stateId"] = state_id
        if project_id:
            input_vars["projectId"] = project_id
        # Auto-assign to active cycle if one exists
        if self._active_cycle_id:
            input_vars["cycleId"] = self._active_cycle_id

        variables = {"input": input_vars}

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
                commit_author=primary_author,
                title=result.title,
            )
            self._state.add_created_issue(record)
            assignee_info = f"assignee={result.assignee or 'unassigned'}"
            project_info = f"project={result.project or 'none'}"
            logger.info(
                f"ISSUE_CREATED | {issue['identifier']} | "
                f"\"{result.title}\" | priority={result.priority} | "
                f"label={result.label} | state={result.state} | "
                f"{assignee_info} | {project_info} | url={issue['url']}"
            )
        else:
            logger.error(f"ISSUE_CREATE_FAILED | {result.title} | {data}")

    def _create_subtask(self, result: GeminiResult, source_shas: list[str], primary_author: str | None = None):
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
        assignee_id = self.resolve_assignee_id(result.assignee)

        # Fallback: if Gemini couldn't match the assignee, resolve from GitHub username
        if not assignee_id and primary_author:
            logger.info(
                f"ASSIGNEE_FALLBACK | Gemini assignee '{result.assignee}' not resolved. "
                f"Trying GitHub username '{primary_author}' → Linear user mapping..."
            )
            assignee_id = self._resolve_github_to_linear_user(primary_author)
            if assignee_id:
                logger.info(f"ASSIGNEE_FALLBACK_OK | '{primary_author}' resolved via GitHub→Linear mapping")

        state_id = self.resolve_state_id(result.state)
        project_id = self.resolve_project_id(result.project)

        mutation = """
        mutation IssueCreate($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue { id identifier url }
          }
        }
        """
        description = result.description + BOT_SIGNATURE

        input_vars = {
            "teamId": self._team_uuid,
            "title": result.title,
            "description": description,
            "priority": result.priority,
            "labelIds": label_ids,
            "parentId": parent_uuid,
        }
        if assignee_id:
            input_vars["assigneeId"] = assignee_id
        if state_id:
            input_vars["stateId"] = state_id
        if project_id:
            input_vars["projectId"] = project_id
        if self._active_cycle_id:
            input_vars["cycleId"] = self._active_cycle_id

        variables = {"input": input_vars}

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
                commit_author=primary_author,
                title=result.title,
            )
            self._state.add_created_issue(record)
            assignee_info = f"assignee={result.assignee or 'unassigned'}"
            project_info = f"project={result.project or 'none'}"
            logger.info(
                f"SUBTASK_CREATED | {issue['identifier']} | "
                f"parent={result.parent_issue_id} | "
                f"\"{result.title}\" | priority={result.priority} | "
                f"label={result.label} | state={result.state} | "
                f"{assignee_info} | {project_info} | url={issue['url']}"
            )
        else:
            logger.error(f"SUBTASK_CREATE_FAILED | parent={result.parent_issue_id} | {result.title} | {data}")

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
        separator = f"\n\n---\n_Updated by **{BOT_NAME}**:_\n"
        new_desc = (existing_desc or "") + separator + result.description

        assignee_id = self.resolve_assignee_id(result.assignee)
        state_id = self.resolve_state_id(result.state)

        mutation = """
        mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) {
            success
            issue { id identifier }
          }
        }
        """
        update_input = {
            "description": new_desc,
            "priority": result.priority,
        }
        if assignee_id:
            update_input["assigneeId"] = assignee_id
        if state_id:
            update_input["stateId"] = state_id

        variables = {
            "id": issue_uuid,
            "input": update_input,
        }

        data = self._gql(mutation, variables)
        update_data = self._safe_get(data, "data", "issueUpdate") or {}

        if update_data.get("success"):
            assignee_info = f"assignee={result.assignee or 'unchanged'}"
            logger.info(
                f"ISSUE_UPDATED | {result.existing_issue_id} | "
                f"\"{result.title}\" | priority={result.priority} | "
                f"label={result.label} | state={result.state} | "
                f"{assignee_info}"
            )
        else:
            logger.error(f"ISSUE_UPDATE_FAILED | {result.existing_issue_id} | {result.title} | {data}")

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
