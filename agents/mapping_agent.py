"""
Mapping Agent
Resolves GitHub usernames to Linear team members using AI-powered inference.

When direct name/email matching fails, this agent gathers contextual evidence
(project membership, recent issue assignees, commit patterns) and asks Gemini
to infer the correct mapping via process of elimination.

Mappings are cached permanently once resolved to avoid repeated lookups.
"""

import json
import time
import logging
import requests
from state import CacheManager
import config

logger = logging.getLogger("agent.mapping")

MAPPING_CACHE_TTL = 86400 * 30  # 30 days — mappings are stable

# Delay between Gemini API calls to avoid rate limits
INTER_CALL_DELAY = 5  # seconds
MAX_RESOLVE_ATTEMPTS = 3

MAPPING_PROMPT = """\
You are a user identity resolution AI. Your job is to match a GitHub username \
to the correct Linear (project management) team member.

You will receive:
1. A GitHub username to identify
2. A list of Linear team members (displayName + email)
3. Contextual clues: which projects the user's commits relate to, who is \
assigned to issues in those projects, and any other available evidence

Your task: determine which Linear team member corresponds to the GitHub user.

Respond with ONLY a valid JSON object:
{
  "linear_display_name": "<exact displayName from the team member list>",
  "confidence": "high" | "medium" | "low",
  "reasoning": "brief explanation of how you matched"
}

Matching strategies (use in order):
1. Name similarity — "FazalulAbid" on GitHub → "Fazalul Abid" on Linear
2. Email prefix match — GitHub username might match the email prefix
3. Project context — if the commit touches Project X, and only member Y works \
on Project X, then the GitHub user is likely member Y
4. Process of elimination — if all other members are accounted for, the \
remaining one must be this user
5. Commit patterns — if the user's commits are in a domain (e.g. mobile, \
backend), match to the member who works in that domain

Rules:
- The linear_display_name MUST be an EXACT displayName from the member list
- If you truly cannot determine the match, set linear_display_name to null \
and confidence to "low"
- Do NOT guess randomly — only match when you have reasonable evidence
- "medium" confidence is acceptable if you have circumstantial evidence
- "high" confidence means strong name/email match or definitive elimination
"""


class MappingAgent:
    """
    Resolves GitHub usernames → Linear team members using Gemini inference.
    Uses project context, issue assignees, and elimination to find matches.
    """

    def __init__(self, gemini_keys: list[str], cache: CacheManager):
        self._keys = gemini_keys
        self._key_index = 0
        self._cache = cache

    def _get_next_key(self) -> str:
        """Rotate through available Gemini API keys."""
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def resolve(
        self,
        github_username: str,
        members_detail: list[dict],
        project_context: list[dict] | None = None,
        commit_context: str | None = None,
        known_mappings: dict[str, str] | None = None,
    ) -> dict | None:
        """
        Attempt to resolve a GitHub username to a Linear team member.

        Args:
            github_username: The GitHub username to resolve
            members_detail: List of {id, displayName, email} for all team members
            project_context: List of {project, assignees} showing who works on what
            commit_context: Description of recent commits by this user
            known_mappings: Already-resolved {github_username: linear_displayName} pairs

        Returns:
            {"linear_user_id": "...", "display_name": "...", "confidence": "..."}
            or None if resolution fails.
        """
        # Check permanent cache first
        cache_key = f"github_mapping:{github_username.lower()}"
        cached = self._cache.get(cache_key, MAPPING_CACHE_TTL)
        if cached:
            logger.info(
                f"MAPPING CACHE HIT | {github_username} → {cached.get('display_name')}"
            )
            return cached

        logger.info(
            f"MAPPING AGENT START | Resolving GitHub user '{github_username}' "
            f"| {len(members_detail)} team members | "
            f"{len(project_context or [])} project context entries"
        )

        prompt = self._build_prompt(
            github_username, members_detail, project_context,
            commit_context, known_mappings
        )

        for attempt in range(MAX_RESOLVE_ATTEMPTS):
            try:
                result = self._call_gemini(prompt)
                if result and result.get("linear_display_name"):
                    display_name = result["linear_display_name"]
                    confidence = result.get("confidence", "low")
                    reasoning = result.get("reasoning", "")

                    # Find the Linear user ID from display name
                    linear_user_id = None
                    for m in members_detail:
                        if m["displayName"] == display_name:
                            linear_user_id = m["id"]
                            break

                    if not linear_user_id:
                        # Try case-insensitive
                        for m in members_detail:
                            if m["displayName"].lower() == display_name.lower():
                                linear_user_id = m["id"]
                                display_name = m["displayName"]  # Use exact casing
                                break

                    if linear_user_id:
                        mapping = {
                            "linear_user_id": linear_user_id,
                            "display_name": display_name,
                            "confidence": confidence,
                        }

                        # Only cache high/medium confidence mappings
                        if confidence in ("high", "medium"):
                            self._cache.set(cache_key, mapping)
                            logger.info(
                                f"MAPPING RESOLVED | {github_username} → "
                                f"{display_name} (confidence={confidence}) | "
                                f"Reason: {reasoning}"
                            )
                        else:
                            logger.info(
                                f"MAPPING TENTATIVE | {github_username} → "
                                f"{display_name} (confidence={confidence}, not cached) | "
                                f"Reason: {reasoning}"
                            )
                        return mapping
                    else:
                        logger.warning(
                            f"Gemini suggested '{display_name}' but it's not "
                            f"a valid team member. Retrying..."
                        )
                else:
                    logger.info(
                        f"MAPPING INCONCLUSIVE | attempt {attempt + 1}/{MAX_RESOLVE_ATTEMPTS} "
                        f"for '{github_username}'"
                    )

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    delay = INTER_CALL_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Rate limited during mapping. Waiting {delay}s "
                        f"(attempt {attempt + 1}/{MAX_RESOLVE_ATTEMPTS})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Mapping API error: {e}")
            except Exception as e:
                logger.error(f"Mapping agent error: {e}")

            # Delay between attempts to avoid rate limits
            if attempt < MAX_RESOLVE_ATTEMPTS - 1:
                time.sleep(INTER_CALL_DELAY)

        logger.warning(f"MAPPING FAILED | Could not resolve '{github_username}' after {MAX_RESOLVE_ATTEMPTS} attempts")
        return None

    def _call_gemini(self, user_prompt: str) -> dict | None:
        """Make a single Gemini API call for mapping resolution."""
        key = self._get_next_key()
        # Use the highest-volume model for mapping to avoid rate issues
        model = "gemini-3.1-flash-lite-preview"
        url = f"{config.GEMINI_API_BASE}/{model}:generateContent?key={key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": MAPPING_PROMPT},
                        {"text": user_prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not text:
            return None

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse mapping response: {e}")
            return None

    def _build_prompt(
        self,
        github_username: str,
        members_detail: list[dict],
        project_context: list[dict] | None,
        commit_context: str | None,
        known_mappings: dict[str, str] | None,
    ) -> str:
        """Build the context prompt for the mapping LLM call."""
        lines = [f"GitHub username to resolve: \"{github_username}\"\n"]

        # Team members
        lines.append("=== LINEAR TEAM MEMBERS ===")
        for m in members_detail:
            email_part = f" ({m.get('email', '')})" if m.get("email") else ""
            lines.append(f"  - {m['displayName']}{email_part}")

        # Known mappings (elimination context)
        if known_mappings:
            lines.append("\n=== ALREADY KNOWN MAPPINGS (these members are accounted for) ===")
            for gh_user, linear_name in known_mappings.items():
                lines.append(f"  - GitHub \"{gh_user}\" → Linear \"{linear_name}\"")
            lines.append(
                "\nUse the above known mappings for elimination. "
                "The unmatched GitHub user is NOT any of these Linear members."
            )

        # Project context
        if project_context:
            lines.append("\n=== PROJECT CONTEXT (who works on what) ===")
            for ctx in project_context:
                project_name = ctx.get("project", "Unknown")
                assignees = ctx.get("assignees", [])
                if assignees:
                    lines.append(
                        f"  - Project \"{project_name}\": "
                        f"assigned to {', '.join(assignees)}"
                    )

        # Commit context
        if commit_context:
            lines.append(f"\n=== RECENT COMMITS BY \"{github_username}\" ===")
            lines.append(f"  {commit_context}")

        lines.append(
            f"\nBased on all the evidence above, which Linear team member "
            f"is the GitHub user \"{github_username}\"?"
        )

        return "\n".join(lines)

    def get_known_mappings(self) -> dict[str, str]:
        """
        Return all cached GitHub → Linear mappings for elimination context.
        """
        cache_data = self._cache._load()
        mappings = {}
        for key, val in cache_data.items():
            if key.startswith("github_mapping:") and val.get("data"):
                gh_user = key.replace("github_mapping:", "")
                display_name = val["data"].get("display_name")
                if display_name:
                    mappings[gh_user] = display_name
        return mappings
