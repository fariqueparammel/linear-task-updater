# GitHub to Linear Sync Engine - Implementation Plan

## 1. Project Overview
A multi-agent Python system that monitors **all repositories in a GitHub organization**, processes commit messages using Gemini AI (with key rotation), and automatically creates or updates tasks in Linear. Each agent is a self-contained module responsible for one concern.

### Safety Principles
- **GitHub: Read-only**. The script only reads commit history. It never writes, pushes, or modifies any repository.
- **Linear: Script-owned tasks only**. Every issue created by the script is tagged with an `auto-sync` label. The script can only update issues it created (tracked via local registry + label check). It **never deletes** issues without explicit user confirmation.
- **No bulk operations**. All mutations are single-issue. No batch deletes, no mass updates.

---

## 2. Architecture: Multi-Agent Modules

```
main.py (Orchestrator)
  │
  ├── config.py           # Env loading, validation, constants
  ├── models.py            # Dataclasses: CommitInfo, GeminiResult, etc.
  ├── state.py             # Persistent state: SHA tracking, buffer, issue registry
  │
  └── agents/
      ├── github_agent.py  # READ-ONLY: Fetches commits from all org repos
      ├── buffer_agent.py  # Manages commit batching (3-commit window + timeout)
      ├── gemini_agent.py  # AI classification with 3-key rotation
      └── linear_agent.py  # SAFE: Creates/updates Linear issues (never deletes)
```

### Data Flow
```
GitHubAgent.fetch_all_org_commits()
    → BufferAgent.add_commits(commits)
    → BufferAgent.get_ready_batch()
    → GeminiAgent.classify(batch)
    → LinearAgent.execute(classification)
    → State.update()
```

---

## 3. Configuration (.env)

```env
# GitHub (Read-Only)
GITHUB_TOKEN=ghp_xxxxxxxxxxxx          # PAT with read:org + repo scope
GITHUB_ORG=your-org-name               # Organization name

# Linear
LINEAR_API_KEY=lin_api_xxxxxxxxxxxx    # Personal API key
LINEAR_TEAM_ID=xxxxxxxx               # Target team ID for issue creation

# Gemini (3 keys for rotation)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...

# Tuning
POLL_INTERVAL_SECONDS=60              # How often to poll (default: 60)
BATCH_SIZE=3                          # Commits per batch (default: 3)
BATCH_TIMEOUT_SECONDS=3600            # Force-process incomplete batch (default: 1hr)
LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR
DRY_RUN=false                         # If true, log actions but don't mutate Linear
```

---

## 4. Module Specifications

### 4A. `config.py` — Configuration
- Load `.env` via `python-dotenv`.
- Validate all required keys exist at startup (fail fast with clear error).
- Export typed constants used by all agents.

### 4B. `models.py` — Data Models
```python
@dataclass
class CommitInfo:
    sha: str
    message: str
    author: str
    repo: str              # "org/repo-name"
    branch: str
    timestamp: str         # ISO format

@dataclass
class GeminiResult:
    action: str            # CREATE_NEW | ADD_SUBTASK | UPDATE_EXISTING
    title: str
    description: str
    priority: int          # 0-4
    label: str             # Bug, Feature, Improvement, Chore, Refactor
    state: str             # Todo, In Progress, Done
    parent_issue_id: str | None
    existing_issue_id: str | None

@dataclass
class LinearIssueRecord:
    issue_id: str          # Linear internal UUID
    identifier: str        # e.g. "TEAM-123"
    url: str
    created_at: str
    source_commits: list[str]  # SHAs that generated this issue
```

### 4C. `state.py` — Persistent State Manager
Manages three state files:

| File | Format | Purpose |
|---|---|---|
| `state/repo_shas.json` | `{"org/repo": "sha123"}` | Last processed SHA per repo |
| `state/commit_buffer.json` | `[CommitInfo, ...]` | Pending commits awaiting batch |
| `state/created_issues.json` | `[LinearIssueRecord, ...]` | Registry of all issues created by this script |

- All reads/writes use atomic file operations (write to temp, then rename).
- State directory is auto-created on first run.

### 4D. `agents/github_agent.py` — GitHub Agent (READ-ONLY)
**Permissions**: Read-only. Uses only GET endpoints. Never writes to any repo.

- **`fetch_org_repos()`**: List all non-archived, non-fork repos in the org via `org.get_repos()`.
- **`fetch_new_commits(repo, last_sha)`**: Get commits from default branch since `last_sha`.
  - If no `last_sha`: take only the latest 1 commit (don't backfill).
  - Filter out: merge commits (`message.startswith("Merge")`), empty messages, `[bot]` authors.
  - Return `List[CommitInfo]` ordered oldest-first.
- **`fetch_all_org_commits(repo_shas)`**: Iterate all repos, call `fetch_new_commits` for each, return aggregated list with updated SHA mapping.

### 4E. `agents/buffer_agent.py` — Buffer Agent
- **`add_commits(commits)`**: Append to buffer, save to disk.
- **`is_ready()`**: Returns `True` if `len(buffer) >= BATCH_SIZE` OR (buffer non-empty AND oldest commit timestamp > `BATCH_TIMEOUT_SECONDS` ago).
- **`get_batch()`**: Pop up to `BATCH_SIZE` commits from front of buffer.
- **`clear_batch(batch)`**: Remove processed commits from buffer, save.

### 4F. `agents/gemini_agent.py` — Gemini Agent
**Key Rotation**: `KeyManager` class with `itertools.cycle` over 3 keys.

- **API**: `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}`
- **System Prompt**:
```
You are a project management AI. You analyze git commit messages and decide
what Linear (project management) action to take.

Respond with ONLY a valid JSON object:
{
  "action": "CREATE_NEW" | "ADD_SUBTASK" | "UPDATE_EXISTING",
  "title": "concise title (max 80 chars)",
  "description": "bullet-point summary of the commits",
  "priority": 0-4 (0=None, 1=Urgent, 2=High, 3=Medium, 4=Low),
  "label": "Bug" | "Feature" | "Improvement" | "Chore" | "Refactor",
  "state": "Todo" | "In Progress" | "Done",
  "parent_issue_id": "TEAM-123 or null",
  "existing_issue_id": "TEAM-456 or null"
}

Rules:
- CREATE_NEW: Commits introduce new work not tied to an existing tracked task.
- ADD_SUBTASK: Commits are clearly part of a larger tracked task. Provide parent_issue_id.
- UPDATE_EXISTING: Commits continue work on an already tracked task. Provide existing_issue_id.
- Default to CREATE_NEW if uncertain.
- For UPDATE_EXISTING and ADD_SUBTASK, only reference issue IDs from this
  list of script-created issues: {created_issues_context}
```

- **Retry Logic**: On 429 → rotate key → wait 2s → retry. Max 3 retries with exponential backoff.
- **JSON Validation**: If response isn't valid JSON, retry once with "respond with valid JSON only".

### 4G. `agents/linear_agent.py` — Linear Agent (SAFE)
**Safety Rules**:
1. Every created issue gets the `auto-sync` label (created on first run if it doesn't exist).
2. Before updating an issue, verify it exists in `created_issues.json` AND has the `auto-sync` label on Linear.
3. **Never delete issues**. No delete mutation exists in this agent.
4. All mutations are single-issue (no batch operations).

**Operations**:

#### `ensure_auto_sync_label()` — Run once on startup
```graphql
# Check if label exists
query { issueLabels(filter: { name: { eq: "auto-sync" } }) { nodes { id name } } }

# Create if missing
mutation { issueLabelCreate(input: { name: "auto-sync", color: "#6B7280", teamId: "..." }) { ... } }
```

#### `fetch_team_labels()` — Cache label name → ID mapping
```graphql
query { issueLabels(filter: { team: { id: { eq: "<TEAM_ID>" } } }) { nodes { id name } } }
```

#### `create_issue(gemini_result)` — For CREATE_NEW
```graphql
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier url }
  }
}
```
- Always includes `auto-sync` label ID in `labelIds`.
- Records the created issue in `created_issues.json`.

#### `create_subtask(gemini_result)` — For ADD_SUBTASK
- Same `issueCreate` mutation with `parentId` set.
- Resolves `parent_issue_id` (e.g. "TEAM-123") to UUID first.
- **Safety check**: parent must be in `created_issues.json`.

#### `update_issue(gemini_result)` — For UPDATE_EXISTING
```graphql
mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) {
    success
    issue { id identifier }
  }
}
```
- **Safety check**: issue must be in `created_issues.json` AND confirmed via Linear API to have `auto-sync` label.
- Appends new description to existing description (doesn't replace).

#### `resolve_issue_identifier(identifier)` — Resolve "TEAM-123" → UUID
```graphql
query { issue(id: "TEAM-123") { id labels { nodes { name } } } }
```

---

## 5. Main Orchestrator (`main.py`)

```python
def main():
    # 1. Init
    config.validate()
    state = StateManager()
    github = GitHubAgent(config.GITHUB_TOKEN, config.GITHUB_ORG)
    buffer = BufferAgent(state)
    gemini = GeminiAgent(config.GEMINI_KEYS)
    linear = LinearAgent(config.LINEAR_API_KEY, config.LINEAR_TEAM_ID, state)

    # 2. One-time setup
    linear.ensure_auto_sync_label()
    linear.fetch_team_labels()

    # 3. Loop
    while True:
        repo_shas = state.get_repo_shas()

        # Fetch
        new_commits, updated_shas = github.fetch_all_org_commits(repo_shas)

        # Buffer
        if new_commits:
            buffer.add_commits(new_commits)
            state.update_repo_shas(updated_shas)

        # Process
        while buffer.is_ready():
            batch = buffer.get_batch()
            result = gemini.classify(batch, state.get_created_issues())

            if result and not config.DRY_RUN:
                linear.execute(result)

            buffer.clear_batch(batch)

        time.sleep(config.POLL_INTERVAL_SECONDS)
```

---

## 6. Per-User Task Correlation

When a new commit arrives, the system correlates it with the commit author's recent Linear tasks to determine whether it's related to existing work or represents new work.

### Flow
```
New commit arrives
    → Extract author (GitHub username)
    → LinearAgent.fetch_user_recent_issues(author) — get user's recent tasks
    → GeminiAgent.classify(batch, created_issues, user_recent_issues)
        → Gemini sees the commit + user's active tasks
        → If related to an existing task → ADD_SUBTASK or UPDATE_EXISTING
        → If unrelated → CREATE_NEW
    → LinearAgent.execute(result)
```

### 6A. User Mapping: GitHub → Linear

The `LinearAgent` queries Linear for team members and builds a mapping of GitHub usernames to Linear user IDs. This mapping is cached on startup and used to fetch per-user issues.

- **`fetch_team_members()`**: Query `team.members` to get `{displayName, email, id}` for all team members.
- **`_github_to_linear_user`**: Dict mapping GitHub username → Linear user ID. Matched by display name (case-insensitive). Falls back to fetching all recent team issues if no user match is found.

### 6B. Per-User Recent Issues

- **`fetch_user_recent_issues(github_username)`**: Given a GitHub username, look up the Linear user ID and fetch their issues updated in the last 14 days.
- Returns a list of `{identifier, title, state}` tuples that Gemini can compare against.
- If the user can't be mapped, falls back to returning the last 10 team-wide issues.

### 6C. Enhanced Gemini Prompt

The Gemini system prompt now includes a new section:

```
The commit author's recent Linear tasks (to check for related work):
  - LAT-45: "Refactor notification system" (In Progress)
  - LAT-42: "Fix login crash on Android 14" (Done)
  ...

If the commit is clearly a continuation of or related to one of the author's
recent tasks, use ADD_SUBTASK (setting parent_issue_id) or UPDATE_EXISTING
(setting existing_issue_id). Only reference issue identifiers from the
script-created issues list OR the author's recent tasks list.
```

### 6D. Critical Task Override

If Gemini determines a commit is **critical** (priority 1 = Urgent), the batch size requirement is overridden:

- The GeminiResult now includes an `is_critical: bool` field.
- When `is_critical` is true, the orchestrator processes the commit immediately — even if the buffer hasn't reached `BATCH_SIZE`.
- The buffer agent exposes `has_critical()` to check if any buffered commit was flagged as potentially critical (commit message contains keywords like "hotfix", "critical", "urgent", "security", "CVE", "crash", "breaking").

### Decision Matrix

| Scenario | Action | Batch Override? |
|---|---|---|
| Commit unrelated to user's tasks | CREATE_NEW | No |
| Commit related to user's active task | ADD_SUBTASK | No |
| Commit continues work on script-owned task | UPDATE_EXISTING | No |
| Any commit flagged as critical/urgent | Any action above | **Yes** — process immediately |

---

## 7. Additional Features

| Feature | Description |
|---|---|
| **Dry-run mode** | Set `DRY_RUN=true` to log all actions without mutating Linear. |
| **Structured logging** | Python `logging` module with configurable level. Logs to console + `logs/sync.log`. |
| **Graceful shutdown** | `SIGINT`/`SIGTERM` handler saves state before exiting. |
| **Per-repo tracking** | Each repo's SHA is tracked independently — no commits are missed across repos. |
| **Created-issues context** | Gemini receives the list of script-created issues so it can suggest `UPDATE_EXISTING` or `ADD_SUBTASK` for known tasks. |
| **Atomic state writes** | Write to temp file then rename — prevents corruption on crash. |
| **Issue ownership guard** | Double-check (local registry + Linear label) before any update. |
| **Per-user correlation** | Each commit is correlated with the author's recent Linear tasks via Gemini. |
| **Critical task override** | Commits with hotfix/critical/urgent keywords bypass batch threshold and process immediately. |
| **Key × Model rotation** | 3 API keys × 5 Gemini models = 15 unique rate limit slots for maximum free-tier throughput. |

---

## 8. Dependencies

```
PyGithub>=2.1.1
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## 9. File Structure

```
linear_update/
├── main.py                    # Orchestrator loop
├── config.py                  # Configuration & validation
├── models.py                  # Dataclasses
├── state.py                   # State persistence
├── agents/
│   ├── __init__.py
│   ├── github_agent.py        # Read-only GitHub poller
│   ├── buffer_agent.py        # Commit batching
│   ├── gemini_agent.py        # AI classification
│   └── linear_agent.py        # Safe Linear mutations
├── state/                     # Auto-created at runtime
│   ├── repo_shas.json
│   ├── commit_buffer.json
│   └── created_issues.json
├── logs/                      # Auto-created at runtime
│   └── sync.log
├── .env                       # Secrets (gitignored)
├── .env.example               # Template
├── requirements.txt
└── plan.md
```
