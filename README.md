## TOTALLY VIBED and havnt Read the code WILL DO IT ASAP
# GitHub to Linear Sync Engine

Automated sync engine that monitors all repositories in a GitHub organization, classifies commits using Gemini AI, and creates/updates tasks in Linear.

## How It Works

```
GitHub Commits → Buffer (3 commits) → Gemini AI Classification → Linear Issue Creation
```

The script runs as a continuous polling loop:

1. **Fetches** new commits from every repo in your GitHub org (read-only)
2. **Buffers** them until 3 commits accumulate (or 1 hour timeout)
3. **Sends** the batch to Gemini AI which decides: create new issue, add subtask, or update existing
4. **Syncs** the result to Linear via GraphQL

## Safety

- **GitHub**: Read-only. Never writes to any repo.
- **Linear**: Only modifies issues it created (tagged `auto-sync`). Never deletes issues.
- **Gemini**: 3-key rotation to avoid rate limits.

## Prerequisites

- Python 3.12+
- GitHub Personal Access Token (with `read:org` + `repo` scope)
- Linear API Key
- 1-3 Gemini API Keys

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
LINEAR_TEAM_ID=your-team-id

# Gemini (at least 1 key required, 3 recommended)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
```

#### Where to get your keys

| Key | Where |
|---|---|
| `GITHUB_TOKEN` | [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens) — enable `read:org` and `repo` scopes |
| `GITHUB_ORG` | Your GitHub organization name (from the URL: `github.com/<org>`) |
| `LINEAR_API_KEY` | [Linear Settings → API → Personal API keys](https://linear.app/settings/api) |
| `LINEAR_TEAM_ID` | Linear Settings → Your Team → copy the ID from the URL |
| `GEMINI_API_KEY_*` | [Google AI Studio → API Keys](https://aistudio.google.com/apikey) |

### 3. Optional tuning (in `.env`)

```env
POLL_INTERVAL_SECONDS=60       # How often to check for new commits
BATCH_SIZE=3                   # Commits per batch sent to Gemini
BATCH_TIMEOUT_SECONDS=3600     # Force-process incomplete batch after 1 hour
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
DRY_RUN=false                  # Set to true to log actions without creating Linear issues
```

## Running

### Option A: Directly with Python

```bash
python main.py
```

Stop with `Ctrl+C` — the script saves state before exiting.

### Option B: Docker

```bash
# Build and run in background
docker compose up -d

# View live logs
docker compose logs -f

# Stop
docker compose down
```

State and logs persist on the host in `./state/` and `./logs/`.

### Option C: Background process (no Docker)

```bash
nohup python -u main.py > output.log 2>&1 &
```

## Running Tests

Validates that all API connections work correctly before running in production.

```bash
python test_integration.py
```

This runs 6 test phases:

| Phase | What it checks |
|---|---|
| 1. Config | All env vars are set |
| 2. GitHub | Can list org repos and fetch commits |
| 3. Gemini | Can classify commits and return valid JSON |
| 4. Linear | Can create/update issues and enforce safety checks |
| 5. Pipeline | Full end-to-end: GitHub → Buffer → Gemini → Linear |
| 6. Models | Offline validation of data models and state management |

Test issues created on Linear are tagged `[TEST]` and `auto-sync`. The test script never auto-deletes — it prints the issue IDs so you can remove them manually.

## Project Structure

```
linear_update/
├── main.py                  # Orchestrator — polling loop
├── config.py                # Environment loading and validation
├── models.py                # Data classes (CommitInfo, GeminiResult, etc.)
├── state.py                 # Persistent state (SHAs, buffer, issue registry)
├── agents/
│   ├── github_agent.py      # Read-only GitHub commit fetcher
│   ├── buffer_agent.py      # Commit batching with timeout
│   ├── gemini_agent.py      # AI classification with key rotation
│   └── linear_agent.py      # Safe Linear issue creation/updates
├── test_integration.py      # Integration test suite
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image
├── docker-compose.yml       # Container orchestration
└── plan.md                  # Architecture documentation
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Missing required environment variables` | Copy `.env.example` to `.env` and fill in all keys |
| GitHub `401 Unauthorized` | Regenerate your `GITHUB_TOKEN` with `read:org` + `repo` scopes |
| GitHub returns 0 repos | Verify `GITHUB_ORG` matches your org name exactly |
| Gemini `429 Too Many Requests` | Add more Gemini API keys (up to 3). The script rotates automatically |
| Linear `Authentication failed` | Check `LINEAR_API_KEY` and `LINEAR_TEAM_ID` are correct |
| No issues being created | Check `logs/sync.log` for errors. Try `DRY_RUN=true` first to see what would happen |
| Script stops after restart | Normal — state is saved in `state/`. It picks up where it left off |
