# CommitPilot — Commit Message Cheat Sheet

## Critical Keywords (Bypass Batch — Immediate Processing)

Include **any** of these keywords anywhere in your commit message to skip the batch queue and trigger immediate Linear task creation:

| Keyword | Use When |
|---|---|
| `hotfix` | Hot fix for production |
| `critical` | Critical severity issue |
| `urgent` | Urgent priority work |
| `security` | Security-related change |
| `CVE-` | CVE identifier (e.g. `CVE-2024-1234`) |
| `crash` | Crash fix |
| `breaking` | Breaking change |
| `emergency` | Emergency fix |
| `rollback` | Rollback / revert |

**Case-insensitive** — `HOTFIX`, `Hotfix`, and `hotfix` all work.

### Examples

```
git commit -m "hotfix: fix payment gateway timeout"
git commit -m "security: patch XSS vulnerability in user input"
git commit -m "urgent: rollback broken migration"
git commit -m "fix CVE-2024-5678 in auth module"
git commit -m "critical crash on startup after config change"
```

All of these bypass the batch threshold (default: 3 commits) and process immediately.

---

## Normal Flow (Batched)

Commits without critical keywords are batched:
- **Batch size**: 3 commits (configurable via `BATCH_SIZE` in `.env`)
- **Timeout**: 1 hour — if a batch doesn't fill up, it processes anyway after 1 hour (`BATCH_TIMEOUT_SECONDS`)

---

## Commits That Are Skipped (Never Create Tasks)

These commit patterns are **automatically filtered out** and will never create a Linear task:

| Pattern | Examples |
|---|---|
| Merge commits | `Merge branch 'feature'`, `Merge pull request #42` |
| Version bumps | `bump`, `release`, `v1.2.3` |
| Trivial / noise | `initial commit`, `first commit`, `init`, `wip`, `tmp` |
| Auto-generated | `auto-commit`, `automated`, `generated` |
| Very short messages | 3 characters or less |
| Bot authors | `dependabot`, `renovate`, `github-actions`, `snyk-bot`, `codecov`, any `[bot]` user |

---

## Guard Agent — Spam Prevention

Even if a commit passes all filters, the Guard Agent will **block** task creation if:

| Check | What Gets Blocked |
|---|---|
| Duplicate SHA | Same commit already created a task |
| Similar title | New task title is >= 82% similar to a task created in the last 24 hours |
| Rate limit | Same author has 5+ tasks created in the last hour |
| Generic title | Title is too vague: "update code", "fix bug", "changes", "wip", "cleanup", etc. |

---

## Writing Good Commit Messages

For best results with CommitPilot's AI classification:

```
<type>: <what you did> in <where>

# Good examples:
feat: add dark mode toggle to settings page
fix: resolve login timeout on slow connections
refactor: migrate payment service to Stripe v3 API
chore: update dependencies for security patches

# Bad examples (may be blocked by Guard):
update code
fix stuff
changes
wip
```
