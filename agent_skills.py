
"""Skill discovery, matching, and instruction loading for MiniAgent."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    instructions: str


def parse_skill(path: Path) -> Skill:
    """Read a Markdown skill with optional YAML-style frontmatter."""
    content = path.read_text(encoding="utf-8")
    metadata = {}

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip().lower()] = value.strip().strip("'\"")

    name = metadata.get("name", path.parent.name)
    description = metadata.get("description", "")

    if not description:
        paragraphs = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.startswith(("#", "---"))
        ]
        description = paragraphs[0] if paragraphs else name

    return Skill(name, description, path, content)


def discover_skills(directory: Path) -> list[Skill]:
    """Recursively discover SKILL.md files."""
    directory = Path(directory)
    if not directory.is_dir():
        return []

    skills = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name.lower() == "skill.md":
            skills.append(parse_skill(path))
    return skills


def tokenize(text: str) -> set[str]:
    """Normalize text for word-overlap matching."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(prompt: str, skill: Skill) -> float:
    """Calculate overlap between task words and skill metadata."""
    task_words = tokenize(prompt)
    skill_words = tokenize(skill.name + " " + skill.description)

    if not task_words or not skill_words:
        return 0.0

    overlap = task_words & skill_words
    return len(overlap) / len(skill_words)


def match_skills(
    prompt: str,
    skills: list[Skill],
    threshold: float = 0.15,
) -> list[tuple[Skill, float]]:
    """Return applicable skills, ranked by similarity."""
    matches = [
        (skill, similarity(prompt, skill))
        for skill in skills
    ]
    return sorted(
        [(skill, score) for skill, score in matches if score >= threshold],
        key=lambda item: (-item[1], item[0].name),
    )


def load_skill(skill: Skill) -> str:
    """Return selected skill instructions for the agent context."""
    return (
        f"Active skill: {skill.name}\n"
        f"Description: {skill.description}\n\n"
        f"{skill.instructions}"
    )
