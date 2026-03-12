# CommitPilot — GitHub to Linear Sync Engine

## 1. Project Overview
A multi-agent Python service (**CommitPilot**) that monitors **all repositories in a GitHub organization**, processes commit messages using Gemini AI (with key × model rotation), and automatically creates or updates tasks in Linear with full workspace-aware field assignment.

All tasks created by CommitPilot are branded with a signature in the description and the `auto-sync` label, making them easily distinguishable from human-created tasks.

### Safety Principles
- **GitHub: Read-only**. The script only reads commit history. Never writes, pushes, or modifies any repository.
- **Linear: Script-owned tasks only**. Every issue gets the `auto-sync` label. Updates are double-checked (local registry + label verification). **Never deletes** issues.
- **No bulk operations**. All mutations are single-issue.

---

## 2. Architecture: Multi-Agent Modules

```
main.py (Orchestrator)
  │
  ├── config.py           # Env loading, validation, constants
  ├── models.py            # Dataclasses: CommitInfo, GeminiResult, LinearIssueRecord
  ├── state.py             # Persistent state + JSON file-based cache with TTL
  │
  └── agents/
      ├── github_agent.py      # READ-ONLY: Fetches commits from all org repos
      ├── buffer_agent.py      # Commit batching + critical commit detection
      ├── gemini_agent.py      # AI classification with key×model rotation + cache cleanup
      ├── linear_agent.py      # SAFE: Creates/updates issues with full field assignment
      ├── mapping_agent.py     # AI-powered GitHub → Linear user resolution
      ├── improvement_agent.py # Self-improving prompt context system
      └── guard_agent.py       # Spam/duplicate detection gate (zero LLM cost)
```

### Data Flow
```
GitHubAgent.fetch_all_org_commits()
    → BufferAgent.add_commits(commits)           [COMMIT_FOUND logged]
    → BufferAgent.is_ready()                     [critical override if hotfix/urgent]
    → LinearAgent.fetch_user_recent_issues()     [per-user correlation]
    → GeminiAgent.classify(batch, context)       [workspace-aware classification]
    → GuardAgent.evaluate(result, shas, issues)  [spam/duplicate gate]
    → LinearAgent.execute(result, shas, author)  [ISSUE_CREATED/UPDATED logged]
    → ImprovementAgent.record_classification()   [accuracy tracking]
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
LINEAR_TEAM_ID=LAT                     # Team key or UUID

# Gemini (unlimited keys — add as many as you want for better rotation)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
# Add more: GEMINI_API_KEY_4, _5, _6, ... (no limit)

# Tuning
POLL_INTERVAL_SECONDS=120
BATCH_SIZE=3
BATCH_TIMEOUT_SECONDS=3600
LOG_LEVEL=INFO
DRY_RUN=false
```

---

## 4. Module Specifications

### 4A. `config.py` — Configuration
- Loads `.env` via `python-dotenv`.
- Validates all required keys at startup (fail fast).
- **Unlimited API keys** for Gemini rotation — dynamically collects all `GEMINI_API_KEY_*` env vars.
- **4 models** (only free-tier models with non-zero RPD):
  - `gemini-3-flash-preview` (5 RPM, 20 RPD)
  - `gemini-3.1-flash-lite-preview` (15 RPM, 500 RPD — best for volume)
  - `gemini-2.5-flash` (5 RPM, 20 RPD)
  - `gemini-2.5-flash-lite` (10 RPM, 20 RPD)
- **N keys × 4 models = N×4 unique rate limit slots**.
- Default poll interval: 120 seconds.

### 4B. `models.py` — Data Models
```python
@dataclass
class CommitInfo:
    sha, message, author, repo, branch, timestamp

@dataclass
class GeminiResult:
    action: str            # CREATE_NEW | ADD_SUBTASK | UPDATE_EXISTING
    title: str
    description: str
    priority: int          # 0=None, 1=Urgent, 2=High, 3=Medium, 4=Low
    label: str             # From workspace valid labels
    state: str             # From workspace valid workflow states
    project: str | None    # From workspace active projects
    parent_issue_id: str | None
    existing_issue_id: str | None
    is_critical: bool      # Hotfix/security/crash flag
    assignee: str | None   # Exact Linear displayName

@dataclass
class LinearIssueRecord:
    issue_id, identifier, url, created_at, source_commits
    commit_author: str | None  # GitHub username of primary commit author
    title: str = ""            # Stored for GuardAgent title similarity checks

@dataclass
class GuardVerdict:
    allowed: bool       # True = pass through, False = blocked
    reason: str         # Human-readable reason (empty if allowed)
    check_name: str     # "sha_duplicate", "title_similarity", "rate_limit", "generic_title", "passed"
```

### 4C. `state.py` — Persistent State + Cache

**StateManager** manages three state files with atomic writes:

| File | Purpose |
|---|---|
| `state/repo_shas.json` | Last processed SHA per repo |
| `state/commit_buffer.json` | Pending commits awaiting batch |
| `state/created_issues.json` | Registry of all issues created by CommitPilot |

**CacheManager** — JSON file-based cache with TTL:

| File | Purpose |
|---|---|
| `state/cache.json` | Cached API lookups with timestamps |

- Entries store `{data, ts}` with per-key TTL enforcement.
- **Auto-purge**: Entries older than 24h removed every 5 minutes.
- **LLM-driven cleanup**: Every hour, sends cache summary to Gemini for intelligent staleness evaluation.
- Cache survives service restarts (file-based, not in-memory).
- TTL defaults: `USER_ISSUES_TTL=600s`, `TEAM_MEMBERS_TTL=3600s`.

### 4D. `agents/github_agent.py` — GitHub Agent (READ-ONLY)
- **`fetch_org_repos()`**: Lists all non-archived, non-fork repos.
- **`fetch_new_commits(repo, last_sha)`**: Gets commits since last SHA, filters merge commits, bot authors, empty messages.
- **`fetch_all_org_commits(repo_shas)`**: Iterates all repos, aggregates new commits.
- **`fetch_commit_author(sha)`**: Looks up a commit author by SHA across the org using GitHub search API. Used during backfill for old issues missing stored author.

### 4E. `agents/buffer_agent.py` — Buffer Agent
- **`add_commits()`**: Appends to buffer, persists to disk.
- **`is_ready()`**: True if `len >= BATCH_SIZE` OR timeout exceeded OR **critical commit detected**.
- **`has_critical()`**: Regex-based detection of hotfix/critical/urgent/security/CVE/crash/breaking/emergency/rollback keywords.
- Critical commits bypass the batch threshold and process immediately.

### 4F. `agents/gemini_agent.py` — Gemini Agent

**SlotManager**: `itertools.product(keys, models)` × `itertools.cycle` for maximum rate limit distribution.

**Classification** (`classify()`):
- Receives: commit batch, created issues, user's recent tasks, full workspace context.
- Workspace context includes: team members, workflow states, labels, projects, cycles.
- System prompt enforces field rules:
  - **Priority**: Integer 0-4 based on commit impact
  - **Label**: Must be from workspace valid labels list
  - **State**: Must be from workspace valid workflow states list
  - **Assignee**: Must be exact displayName from team member list, matched via name similarity, email prefix, or elimination
  - **Project**: Optional, from active projects if applicable
- Learned rules from ImprovementAgent are appended to the system prompt as `=== LEARNED RULES ===`.
- Retry logic: 8 retries with exponential backoff (3s base, 60s cap). Rotates to next slot on 429 or error.

**Cache Cleanup** (`evaluate_cache_cleanup()`):
- Sends cache entry summary (key, age, data type) to Gemini.
- Gemini decides which entries are stale based on rules (user_issues >30min, team_members >2h, anything >6h).
- Returns list of keys to purge.

### 4G. `agents/linear_agent.py` — Linear Agent (SAFE)

**Workspace Setup** (run once on startup, each step wrapped in try/except):
- `resolve_team()` — Resolves team key (e.g. "LAT") to UUID
- `ensure_auto_sync_label()` — Creates `auto-sync` label if missing
- `fetch_team_labels()` — Caches label name → ID
- `fetch_workflow_states()` — Caches state name → ID (e.g. "In Progress", "Todo", "Done")
- `fetch_projects()` — Caches active project name → ID
- `fetch_cycles()` — Caches cycles, auto-detects current active cycle
- `fetch_team_members()` — Caches displayName/email → user ID, stores full member details for Gemini context

**Workspace Context** (`get_workspace_context()`):
Returns `{members, workflow_states, labels, projects, cycles, has_active_cycle}` for Gemini prompt.

**GitHub → Linear User Resolution** (`_resolve_github_to_linear_user()`):
1. Direct lowercase match in user cache
2. Substring / fuzzy match
3. Exact displayName match from member details
4. **MappingAgent** fallback: AI-powered resolution with project context + elimination

**Field Resolution**:
- `resolve_state_id()` — State name → UUID (case-insensitive)
- `resolve_project_id()` — Project name → UUID (case-insensitive)
- `resolve_assignee_id()` — Display name → user UUID (multi-strategy)

**Mutations** — All include: `assigneeId`, `stateId`, `projectId`, `cycleId` (auto-assigned to active cycle), `labelIds` (always includes `auto-sync`), `BOT_SIGNATURE` in description, `commit_author` stored in record.

**Backfill** (`backfill_created_issues()`):
- Runs on startup to fix previously created tasks missing assignee/state/cycle.
- **Assignee backfill**: For unassigned issues:
  1. Uses stored `commit_author` if available
  2. Falls back to GitHub search API (`fetch_commit_author(sha)`) to look up the commit author from source SHAs
  3. Resolves GitHub username → Linear member via MappingAgent
  4. Updates the issue with the resolved `assigneeId`
- Moves Backlog items to Todo.
- Assigns active cycle to uncycled issues.

**Branding**:
- `BOT_NAME = "CommitPilot"`
- All descriptions include: `_Created by **CommitPilot** — automated commit-to-task sync_`
- Update descriptions include: `_Updated by **CommitPilot**:_`

### 4H. `agents/mapping_agent.py` — Mapping Agent

AI-powered GitHub → Linear user resolution when direct name/email matching fails.

**Resolution Flow**:
1. Check permanent cache (30-day TTL) for known mapping
2. Gather context: team members, project assignees, known mappings
3. Send to Gemini with specialized mapping prompt
4. Gemini uses: name similarity → email prefix → project context → process of elimination
5. Cache high/medium confidence results permanently
6. Update `_user_cache` for fast future lookups

**Features**:
- Uses `gemini-3.1-flash-lite-preview` (highest RPD) to avoid rate issues
- 3 retry attempts with exponential backoff
- 5s delay between API calls to respect rate limits
- Known mappings passed as elimination context (already-matched users are excluded)
- `fetch_project_assignees()` in LinearAgent provides who-works-on-what context

### 4I. `agents/improvement_agent.py` — Self-Improving Context Agent

Tracks classification accuracy and iteratively improves the Gemini prompt context. Improvements are additive — the baseline context is NEVER deleted.

**Promotion Threshold**: An improved context is only promoted to "active" after **20 consecutive correct** classifications.

**Tracking** (`record_classification()`):
- Records every classification result (correct/incorrect, field values)
- Maintains accuracy statistics, error patterns, consecutive correct streak
- Keeps last 100 classifications and last 50 error patterns
- Automatically checks for promotion after each record

**Improvement Generation** (`generate_improvement()`):
- Runs every 2 hours if there are recent errors
- Sends current system prompt + error patterns + recent correct examples to Gemini
- Generates targeted, conservative improvements (small rule changes)
- Saves as **candidate** (NOT active) — needs validation first
- Only one candidate at a time

**Promotion** (`_promote_candidate()`):
- After 20 consecutive correct classifications with the candidate active
- Promoted improvements are saved to history (never deleted)
- Active improvements are appended as `=== LEARNED RULES (validated through usage) ===` to the system prompt

**Cache Keys**:
- `improvement:tracker` — Accuracy stats and classification history
- `improvement:candidate` — Current improvement candidate (7-day TTL)
- `improvement:history` — All promoted improvements (permanent)

### 4J. `agents/guard_agent.py` — Guard Agent (Zero-Cost Gate)

Lightweight spam/duplicate detection agent that runs between Gemini classification and Linear execution. Uses ZERO LLM calls — pure string matching and rate checks.

**Only evaluates CREATE_NEW actions** — ADD_SUBTASK and UPDATE_EXISTING always pass.

**Checks (in order)**:

| # | Check | What It Detects | Mechanism |
|---|---|---|---|
| 1 | SHA Duplicate | Same commit already created an issue | Set lookup against `created_issues[].source_commits` |
| 2 | Title Similarity | Near-duplicate issue title | `difflib.SequenceMatcher` (threshold >= 0.82) against issues created in last 24h + user's recent issues |
| 3 | Rate Limit | Author creating too many issues too fast | Count author's issues in last 1 hour (max 5) |
| 4 | Generic Title | Vague/noise titles ("update code", "fix bug") | Regex pattern matching against known generic patterns + min 8 char length |

**Thresholds**:
- `TITLE_SIMILARITY_THRESHOLD = 0.82`
- `TITLE_SIMILARITY_WINDOW = 86400` (24 hours)
- `RATE_LIMIT_MAX_ISSUES = 5` per author
- `RATE_LIMIT_WINDOW = 3600` (1 hour)
- `MIN_TITLE_LENGTH = 8` characters

**Returns**: `GuardVerdict(allowed, reason, check_name)` — blocks logged as `GUARD_BLOCKED | check=... | reason=... | title=...`

---

## 5. Orchestrator (`main.py`)

```python
def main():
    # 1. Config & logging
    # 2. Init agents: StateManager, CacheManager, GitHubAgent, BufferAgent,
    #    GeminiAgent, LinearAgent, MappingAgent, ImprovementAgent, GuardAgent
    #    - linear.set_mapping_agent(mapping)
    #    - linear.set_github_agent(github)
    #    - gemini.set_improvement_agent(improvement)
    #    - guard = GuardAgent(state)
    # 3. One-time setup (each step wrapped in try/except):
    #    - resolve_team, ensure_auto_sync_label, fetch_team_labels
    #    - fetch_workflow_states, fetch_projects, fetch_cycles, fetch_team_members
    # 4. Backfill: resolve & assign unassigned issues, fix missing cycle/state
    # 5. Main loop (each step independently error-handled):
    #    A. Fetch new commits → log COMMIT_FOUND events
    #    B. Buffer commits (critical override for hotfix/urgent)
    #    C. Process batches:
    #       - Per-user correlation: fetch author's recent Linear tasks
    #       - Classify with full workspace context
    #       - Guard check: block spam/duplicates (zero LLM cost)
    #       - Execute: create/update/subtask with all fields + commit_author stored
    #       - Track classification for self-improvement
    #       - Always clear batch in finally block (prevents infinite retry)
    #    D. Periodic self-improvement (every 2 hours)
    #    E. Periodic LLM-driven cache cleanup (every hour)
    #    F. Periodic workspace refresh (every 30 min, configurable)
    #       - Re-fetches labels, workflow states, projects, cycles, members
    #       - Picks up new items added in Linear since startup
    #    - Sleep with graceful shutdown support
```

### Error Handling Strategy

Every sub-step in the main loop is independently wrapped so a single failure never crashes the service:

| Error Tag | Scope | Impact |
|---|---|---|
| `FETCH_COMMITS_ERROR` | GitHub API call | Skips to next cycle, no commits lost |
| `BUFFER_ERROR` | Commit buffering | Skips buffering, commits may be re-fetched |
| `USER_ISSUES_ERROR` | User correlation fetch | Classification proceeds without user context |
| `BATCH_PROCESS_ERROR` | Batch classification | Batch is **always cleared** via `finally` to prevent infinite loops |
| `LINEAR_EXECUTE_ERROR` | Linear mutation | Improvement tracking still runs |
| `IMPROVEMENT_TRACK_ERROR` | Accuracy recording | Non-critical, skipped silently |
| `IMPROVEMENT_ERROR` | Prompt improvement generation | Non-critical, retried next interval |
| `CACHE_CLEANUP_ERROR` | LLM cache review | Non-critical, retried next interval |
| `SETUP_ERROR` | Individual startup step | Other setup steps still run |
| `BACKFILL_STARTUP_ERROR` | Backfill on startup | Main loop starts regardless |
| `GUARD_BLOCKED` | Spam/duplicate detected | Issue creation skipped, batch cleared |
| `WORKSPACE_REFRESH_ERROR` | Workspace data refresh | Individual refresh step fails, others still run |
| `MAIN_LOOP_ERROR` | Unexpected outer error | Caught, logged, loop continues |

---

## 6. Per-User Task Correlation

### Flow
```
New commit arrives
    → Extract author (GitHub username)
    → _resolve_github_to_linear_user(author)
        → Direct match? → Use it
        → Substring match? → Use it
        → Fail? → MappingAgent.resolve(author, members, project_context)
            → Gemini infers via elimination / project context
            → Cache permanently for future lookups
    → LinearAgent.fetch_user_recent_issues(author)
    → GeminiAgent.classify(batch, created_issues, user_recent_issues, workspace_context)
        → If related to existing task → ADD_SUBTASK or UPDATE_EXISTING
        → If unrelated → CREATE_NEW
```

### Critical Task Override

| Scenario | Action | Batch Override? |
|---|---|---|
| Commit unrelated to user's tasks | CREATE_NEW | No |
| Commit related to user's active task | ADD_SUBTASK | No |
| Commit continues work on script-owned task | UPDATE_EXISTING | No |
| Commit contains hotfix/critical/urgent/security/CVE/crash/breaking keywords | Any action | **Yes** — process immediately |

---

## 7. Event Logging

All significant events are logged with distinctive uppercase tags for easy filtering:

| Event | Log Tag |
|---|---|
| Commit discovered | `COMMIT_FOUND \| repo=... \| author=... \| sha=... \| "message"` |
| Task created | `ISSUE_CREATED \| LAT-xxx \| "title" \| priority=... \| label=... \| state=... \| assignee=... \| project=... \| url=...` |
| Subtask created | `SUBTASK_CREATED \| LAT-xxx \| parent=LAT-yyy \| ...` |
| Task updated | `ISSUE_UPDATED \| LAT-xxx \| "title" \| ...` |
| Create failed | `ISSUE_CREATE_FAILED \| ...` |
| Update failed | `ISSUE_UPDATE_FAILED \| ...` |
| User mapping resolved | `MAPPING RESOLVED \| github_user → display_name (confidence=high)` |
| Mapping failed | `MAPPING FAILED \| Could not resolve 'github_user'` |
| Backfill started | `BACKFILL_START \| Checking N issue(s)` |
| Backfill assignee resolved | `BACKFILL_ASSIGNEE \| LAT-xxx \| github=user → linear_user=...` |
| Backfill completed | `BACKFILL_COMPLETE \| Updated N/M issue(s)` |
| Guard blocked | `GUARD_BLOCKED \| check=... \| reason=... \| title=...` |
| Improvement candidate | `IMPROVEMENT CANDIDATE GENERATED \| N suggestion(s)` |
| Context promoted | `CONTEXT PROMOTED \| Version N \| Validated with 20+ consecutive correct` |
| Cache cleanup | `LLM cache cleanup: removed N entries: [keys]` |
| Workspace refresh | `WORKSPACE_REFRESH \| Refreshing projects, states, labels, cycles, members...` |
| Workspace refresh done | `WORKSPACE_REFRESH_DONE \| N members, N states, N labels, N projects` |

---

## 8. Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/*
COPY config.py models.py state.py main.py ./
COPY agents/ ./agents/
COPY entrypoint.sh ./
RUN mkdir -p /app/state /app/logs
VOLUME ["/app/state", "/app/logs"]
RUN useradd --create-home appuser && chown -R appuser:appuser /app
ENTRYPOINT ["./entrypoint.sh"]
```

### entrypoint.sh (Permission Fix + Privilege Drop)
```bash
#!/bin/sh
chown -R appuser:appuser /app/state /app/logs 2>/dev/null || true
exec gosu appuser python -u main.py
```

### docker-compose.yml
```yaml
services:
  sync-engine:
    build: .
    container_name: linear-sync-engine
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./state:/app/state
      - ./logs:/app/logs
    develop:
      watch:
        - action: rebuild
          path: .
          ignore:
            - state/
            - logs/
            - .env
            - "*.md"
            - .git/
```

### Development with auto-rebuild
```bash
# Auto-rebuild on code changes (watches source files, ignores state/logs/.env)
docker compose watch

# Manual rebuild (production)
docker compose up -d --build
```

---

## 9. Additional Features

| Feature | Description |
|---|---|
| **CommitPilot branding** | All created tasks show `_Created by **CommitPilot**_` in description + `auto-sync` label |
| **Workspace-aware classification** | Gemini receives all valid labels, states, projects, members, cycles from Linear |
| **AI-powered user mapping** | MappingAgent uses project context + elimination to resolve unknown GitHub users |
| **Backfill with assignee resolution** | On startup, unassigned issues get their commit author resolved via stored author or GitHub SHA lookup, then mapped to Linear member |
| **Self-improving prompts** | ImprovementAgent tracks accuracy, generates prompt improvements, validates with 20 consecutive correct before promotion |
| **Auto cycle assignment** | All new issues auto-assigned to the current active Linear cycle |
| **Commit author tracking** | `commit_author` stored in `LinearIssueRecord` for future backfill without GitHub lookups |
| **File-based caching** | JSON cache in `state/cache.json` survives restarts, with TTL and auto-purge |
| **LLM-driven cache cleanup** | Gemini periodically reviews cache entries and purges stale data |
| **Critical task override** | Hotfix/security/urgent commits bypass batch threshold |
| **Per-user correlation** | Commits correlated with author's recent Linear tasks |
| **Key × Model rotation** | N API keys × 4 models = N×4 unique rate limit slots (unlimited keys) |
| **Resilient main loop** | Every sub-step independently error-handled; batch always cleared in `finally`; setup failures don't block startup |
| **Docker Compose Watch** | `docker compose watch` auto-rebuilds on source file changes |
| **Dry-run mode** | `DRY_RUN=true` logs actions without mutating Linear |
| **Graceful shutdown** | SIGINT/SIGTERM saves state before exiting |
| **Atomic state writes** | Write to temp file → rename to prevent corruption |
| **Spam/duplicate guard** | GuardAgent blocks duplicate commits, similar titles, rate limit violations, and generic titles — zero LLM cost |
| **Periodic workspace refresh** | Projects, states, labels, cycles, members re-fetched every 30 min to detect new items |
| **Issue ownership guard** | Double-check (local registry + Linear label) before any update |

---

## 10. File Structure

```
linear_update/
├── main.py                    # Orchestrator loop (resilient error handling)
├── config.py                  # Configuration & validation
├── models.py                  # Dataclasses (CommitInfo, GeminiResult, LinearIssueRecord, GuardVerdict)
├── state.py                   # State persistence + CacheManager
├── Dockerfile
├── docker-compose.yml         # Includes develop.watch for auto-rebuild
├── entrypoint.sh              # Permission fix + privilege drop
├── agents/
│   ├── __init__.py
│   ├── github_agent.py        # Read-only GitHub poller + commit author lookup
│   ├── buffer_agent.py        # Commit batching + critical detection
│   ├── gemini_agent.py        # AI classification + cache cleanup + learned rules
│   ├── linear_agent.py        # Safe Linear mutations + backfill with assignee resolution
│   ├── mapping_agent.py       # AI-powered user mapping
│   ├── improvement_agent.py   # Self-improving prompt context system
│   └── guard_agent.py         # Spam/duplicate detection gate (zero LLM cost)
├── state/                     # Auto-created at runtime
│   ├── repo_shas.json
│   ├── commit_buffer.json
│   ├── created_issues.json
│   └── cache.json             # Persistent cache (mappings, improvements, user issues)
├── logs/
│   └── sync.log
├── .env                       # Secrets (gitignored)
├── .env.example
├── requirements.txt
├── cheat_sheet.md             # Critical keyword quick reference
├── plan.md
└── test_integration.py
```
