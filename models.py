from dataclasses import dataclass, field, asdict


@dataclass
class CommitInfo:
    sha: str
    message: str
    author: str
    repo: str          # "org/repo-name"
    branch: str
    timestamp: str     # ISO format

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CommitInfo":
        return cls(**d)


@dataclass
class GeminiResult:
    action: str                        # CREATE_NEW | ADD_SUBTASK | UPDATE_EXISTING
    title: str
    description: str
    priority: int                      # 0-4
    label: str                         # Bug, Feature, Improvement, Chore, Refactor
    state: str                         # Todo, In Progress, Done
    project: str | None = None             # Project name chosen by Gemini
    parent_issue_id: str | None = None
    existing_issue_id: str | None = None
    is_critical: bool = False
    assignee: str | None = None            # Linear display name chosen by Gemini

    VALID_ACTIONS = {"CREATE_NEW", "ADD_SUBTASK", "UPDATE_EXISTING"}
    VALID_LABELS = {"Bug", "Feature", "Improvement", "Chore", "Refactor"}
    VALID_STATES = {"Todo", "In Progress", "Done"}

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.action not in self.VALID_ACTIONS:
            errors.append(f"Invalid action: {self.action}")
        if not self.title or len(self.title) > 120:
            errors.append(f"Title must be 1-120 chars, got {len(self.title or '')}")
        if not isinstance(self.priority, int) or not 0 <= self.priority <= 4:
            errors.append(f"Priority must be 0-4, got {self.priority}")
        if self.action == "ADD_SUBTASK" and not self.parent_issue_id:
            errors.append("ADD_SUBTASK requires parent_issue_id")
        if self.action == "UPDATE_EXISTING" and not self.existing_issue_id:
            errors.append("UPDATE_EXISTING requires existing_issue_id")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "GeminiResult":
        return cls(
            action=d.get("action", "CREATE_NEW"),
            title=d.get("title", ""),
            description=d.get("description", ""),
            priority=int(d.get("priority", 3)),
            label=d.get("label", "Chore"),
            state=d.get("state", "Todo"),
            project=d.get("project"),
            parent_issue_id=d.get("parent_issue_id"),
            existing_issue_id=d.get("existing_issue_id"),
            is_critical=bool(d.get("is_critical", False)),
            assignee=d.get("assignee"),
        )


@dataclass
class LinearIssueRecord:
    issue_id: str       # Linear internal UUID
    identifier: str     # e.g. "TEAM-123"
    url: str
    created_at: str
    source_commits: list[str] = field(default_factory=list)
    commit_author: str | None = None  # GitHub username of the primary commit author

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LinearIssueRecord":
        return cls(
            issue_id=d["issue_id"],
            identifier=d["identifier"],
            url=d["url"],
            created_at=d["created_at"],
            source_commits=d.get("source_commits", []),
            commit_author=d.get("commit_author"),
        )
