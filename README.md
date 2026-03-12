# CommitPilot — GitHub to Linear Sync Engine

Automated multi-agent service that monitors all repositories in a GitHub organization, classifies commits using Gemini AI, and creates/updates tasks in Linear with full workspace-aware field assignment.

## How It Works

```
GitHub Commits → Buffer → Gemini AI Classification → Guard (spam check) → Linear Issue Creation
                           (6 keys × 4 models)        (zero LLM cost)      (assignee, state, project, cycle, label, priority)
```

1. **Fetches** new commits from every repo in your GitHub org (read-only)
2. **Buffers** them until 3 commits accumulate (or 1 hour timeout). Critical commits (hotfix/security) process immediately
3. **Classifies** the batch with Gemini AI using full workspace context (team members, labels, states, projects, cycles)
4. **Guards** against spam — blocks duplicate commits, similar titles, rate limit violations, and generic titles (zero LLM cost)
5. **Creates/updates** Linear issues with all fields resolved: assignee, workflow state, project, cycle, labels, priority
6. **Self-improves** — tracks classification accuracy and iteratively refines the AI prompt after 20 consecutive correct results

## Architecture

```
main.py (Orchestrator)
  ├── GitHubAgent      Read-only commit fetcher + commit author lookup
  ├── BufferAgent      Commit batching + critical commit detection
  ├── GeminiAgent      AI classification with key×model rotation + cache cleanup
  ├── GuardAgent       Spam/duplicate detection gate (zero LLM cost)
  ├── LinearAgent      Safe Linear mutations + backfill + user resolution
  ├── MappingAgent     AI-powered GitHub → Linear user resolution
  └── ImprovementAgent Self-improving prompt context system
```

## Safety

- **GitHub**: Read-only. Never writes to any repo
- **Linear**: Only modifies issues it created (tagged `auto-sync`). Never deletes issues. Double-checked before every update (local registry + label verification)
- **Gemini**: 6-key × 4-model rotation (24 rate limit slots). Exponential backoff on 429s
- **Resilient**: Every step independently error-handled — a single failure never crashes the service

## Prerequisites

- Python 3.12+
- GitHub Personal Access Token (with `read:org` + `repo` scope)
- Linear API Key
- 1+ Gemini API Keys (more keys = better rate limit distribution)

## Setup

### 1. Clone and install dependencies

```bash
cd linear_update
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# GitHub (Read-Only)
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_ORG=your-org-name

# Linear
LINEAR_API_KEY=lin_api_xxxxxxxxxxxx
LINEAR_TEAM_ID=LAT                    # Team key (e.g. "LAT") or UUID

# Gemini (at least 1 key required, add as many as you want for better rotation)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
# Add more keys: GEMINI_API_KEY_4, GEMINI_API_KEY_5, ... (no limit)
```

#### Where to get your keys

| Key | Where |
|---|---|
| `GITHUB_TOKEN` | [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens) — enable `read:org` and `repo` scopes |
| `GITHUB_ORG` | Your GitHub organization name (from the URL: `github.com/<org>`) |
| `LINEAR_API_KEY` | [Linear Settings → API → Personal API keys](https://linear.app/settings/api) |
| `LINEAR_TEAM_ID` | Your team key (e.g. `LAT`) visible in issue identifiers like `LAT-123` |
| `GEMINI_API_KEY_*` | [Google AI Studio → API Keys](https://aistudio.google.com/apikey) — add as many as you want (`_1`, `_2`, `_3`, ...) |

### 3. Optional tuning (in `.env`)

```env
POLL_INTERVAL_SECONDS=120     # How often to check for new commits (default: 120)
BATCH_SIZE=3                  # Commits per batch sent to Gemini
BATCH_TIMEOUT_SECONDS=3600    # Force-process incomplete batch after 1 hour
WORKSPACE_REFRESH_SECONDS=1800 # Re-fetch projects, states, labels etc. (default: 30 min)
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
DRY_RUN=false                 # Set to true to log actions without creating Linear issues
```

## Running

### Option A: Docker (recommended)

```bash
# Build and run in background
docker compose up -d --build

# View live logs
docker compose logs -f

# Stop
docker compose down
```

### Option B: Docker with auto-rebuild on code changes (development)

```bash
# Watches source files and auto-rebuilds on changes
docker compose watch
```

This monitors all source files (ignoring `state/`, `logs/`, `.env`, `.md`, `.git/`) and automatically rebuilds + restarts the container when you make changes.

### Option C: Directly with Python

```bash
python main.py
```

Stop with `Ctrl+C` — the script saves state before exiting.

### Option D: Background process (no Docker)

```bash
nohup python -u main.py > output.log 2>&1 &
```

State and logs persist on the host in `./state/` and `./logs/`.

## What Happens on Startup

1. Validates all environment variables
2. Resolves Linear team, fetches workspace metadata (labels, states, projects, cycles, members)
3. **Backfill**: Checks previously created issues and fixes missing fields:
   - Resolves commit author → Linear member for unassigned issues
   - Assigns active cycle to uncycled issues
   - Moves Backlog items to Todo
4. Starts the polling loop

## Key Features

| Feature | Description |
|---|---|
| **Workspace-aware classification** | Gemini receives all valid labels, states, projects, members, cycles from Linear — returns values that actually exist |
| **AI-powered user mapping** | When direct name matching fails, MappingAgent uses project context + process of elimination to resolve GitHub users to Linear members |
| **Self-improving prompts** | ImprovementAgent tracks accuracy, generates prompt fixes from errors, validates with 20 consecutive correct before promotion |
| **Backfill with assignee resolution** | On startup, unassigned issues get their commit author looked up and mapped to a Linear member |
| **Spam/duplicate guard** | GuardAgent blocks duplicate commits, similar titles, rate limit violations, and generic titles before Linear execution — zero LLM cost |
| **Critical commit override** | Commits containing hotfix/security/urgent/CVE/crash keywords bypass batch threshold and process immediately (see [`cheat_sheet.md`](cheat_sheet.md)) |
| **Per-user task correlation** | Commits are correlated with the author's recent Linear tasks to decide CREATE_NEW vs ADD_SUBTASK vs UPDATE_EXISTING |
| **Key × Model rotation** | N API keys × 4 models = N×4 unique rate limit slots before any repeat (unlimited keys supported) |
| **Resilient error handling** | Every sub-step independently wrapped — a GitHub API failure doesn't block Linear updates, a Gemini timeout doesn't crash the loop |
| **Periodic workspace refresh** | Projects, workflow states, labels, cycles, and members are re-fetched every 30 min to pick up new items (`WORKSPACE_REFRESH_SECONDS`) |
| **LLM-driven cache cleanup** | Gemini periodically reviews cached data and purges stale entries |

## Log Tags

All significant events use distinctive uppercase tags for easy filtering:

```
COMMIT_FOUND    | New commit detected
ISSUE_CREATED   | Linear task created
SUBTASK_CREATED | Linear subtask created
ISSUE_UPDATED   | Existing task updated
BACKFILL_START  | Backfill check started
BACKFILL_ASSIGNEE | Author resolved for unassigned issue
BACKFILL_COMPLETE | Backfill finished
MAPPING RESOLVED  | GitHub → Linear user mapping found
GUARD_BLOCKED   | Spam/duplicate detected, issue creation skipped
WORKSPACE_REFRESH | Workspace data re-fetched (projects, states, labels, cycles, members)
```

Error tags: `FETCH_COMMITS_ERROR`, `BATCH_PROCESS_ERROR`, `LINEAR_EXECUTE_ERROR`, `ISSUE_CREATE_FAILED`, `ISSUE_UPDATE_FAILED`

## Critical Commit Keywords

Include any of these keywords in your commit message to bypass batch queuing and trigger immediate processing:

`hotfix` · `critical` · `urgent` · `security` · `CVE-` · `crash` · `breaking` · `emergency` · `rollback`

Case-insensitive. See [`cheat_sheet.md`](cheat_sheet.md) for full details including skip patterns and writing tips.

## Running Tests

```bash
python test_integration.py
```

| Phase | What it checks |
|---|---|
| 1. Config | All env vars are set |
| 2. GitHub | Can list org repos and fetch commits |
| 3. Gemini | Can classify commits and return valid JSON |
| 4. Linear | Can create/update issues and enforce safety checks |
| 5. Pipeline | Full end-to-end: GitHub → Buffer → Gemini → Linear |
| 6. Models | Offline validation of data models and state management |

Test issues created on Linear are tagged `[TEST]` and `auto-sync`.

## Project Structure

```
linear_update/
├── main.py                    # Orchestrator — resilient polling loop
├── config.py                  # Environment loading and validation
├── models.py                  # Data classes (CommitInfo, GeminiResult, LinearIssueRecord, GuardVerdict)
├── state.py                   # Persistent state + JSON cache with TTL
├── agents/
│   ├── github_agent.py        # Read-only GitHub commit fetcher + author lookup
│   ├── buffer_agent.py        # Commit batching + critical detection
│   ├── gemini_agent.py        # AI classification with key×model rotation
│   ├── linear_agent.py        # Safe Linear mutations + backfill + user resolution
│   ├── mapping_agent.py       # AI-powered GitHub → Linear user mapping
│   ├── improvement_agent.py   # Self-improving prompt context system
│   └── guard_agent.py         # Spam/duplicate detection gate (zero LLM cost)
├── state/                     # Auto-created, persists across restarts
│   ├── repo_shas.json         # Last processed SHA per repo
│   ├── commit_buffer.json     # Pending commits
│   ├── created_issues.json    # Registry of all CommitPilot-created issues
│   └── cache.json             # Cached API lookups, mappings, improvements
├── logs/
│   └── sync.log
├── test_integration.py        # Integration test suite
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
├── Dockerfile
├── docker-compose.yml         # With develop.watch for auto-rebuild
├── entrypoint.sh              # Permission fix + privilege drop
├── cheat_sheet.md             # Critical keyword quick reference
└── plan.md                    # Detailed architecture documentation
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Missing required environment variables` | Copy `.env.example` to `.env` and fill in all keys |
| GitHub `401 Unauthorized` | Regenerate your `GITHUB_TOKEN` with `read:org` + `repo` scopes |
| GitHub returns 0 repos | Verify `GITHUB_ORG` matches your org name exactly |
| Gemini `429 Too Many Requests` | Add more Gemini API keys (up to 6). The script rotates through all key×model combinations |
| Gemini `404 Not Found` | A model may have been deprecated. Check `config.py` for current model list |
| Linear `Authentication failed` | Check `LINEAR_API_KEY` and `LINEAR_TEAM_ID` are correct |
| No issues being created | Check `logs/sync.log` for errors. Try `DRY_RUN=true` first to see what would happen |
| Assignees not being set | Verify team members exist in Linear and Gemini can match GitHub usernames to display names |
| Script stops after restart | Normal — state is saved in `state/`. It picks up where it left off |
| Container not picking up code changes | Use `docker compose watch` instead of `docker compose up` for auto-rebuild |
