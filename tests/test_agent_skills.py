
"""Tests for agent skill discovery, matching, and loading."""

from pathlib import Path

from agent_skills import (
    discover_skills,
    load_skill,
    match_skills,
    parse_skill,
    similarity,
)


def make_skill(tmp_path, name="validate-robot"):
    folder = tmp_path / name
    folder.mkdir()

    path = folder / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "description: Validate robot Q-learning code and tests\n"
        "---\n\n"
        "# Instructions\n"
        "Run the robot validation script and inspect failures.\n",
        encoding="utf-8",
    )
    return path


def test_parse_skill(tmp_path):
    path = make_skill(tmp_path)
    skill = parse_skill(path)

    assert skill.name == "validate-robot"
    assert "Q-learning" in skill.description
    assert "validation script" in skill.instructions


def test_discover_skills(tmp_path):
    make_skill(tmp_path)
    skills = discover_skills(tmp_path)

    assert len(skills) == 1
    assert skills[0].name == "validate-robot"


def test_discover_missing_directory(tmp_path):
    assert discover_skills(tmp_path / "missing") == []


def test_case_insensitive_filename(tmp_path):
    path = make_skill(tmp_path)
    path.rename(path.with_name("skill.md"))

    assert len(discover_skills(tmp_path)) == 1


def test_relevant_prompt_matches(tmp_path):
    skill = parse_skill(make_skill(tmp_path))
    prompt = "Validate my robot Q-learning code and tests"

    assert similarity(prompt, skill) >= 0.15
    assert match_skills(prompt, [skill])


def test_irrelevant_prompt_does_not_match(tmp_path):
    skill = parse_skill(make_skill(tmp_path))
    prompt = "Explain the history of medieval architecture"

    assert similarity(prompt, skill) == 0
    assert match_skills(prompt, [skill]) == []


def test_load_skill(tmp_path):
    skill = parse_skill(make_skill(tmp_path))
    loaded = load_skill(skill)

    assert "Active skill: validate-robot" in loaded
    assert "Run the robot validation script" in loaded


def test_similarity_empty_prompt(tmp_path):
    skill = parse_skill(make_skill(tmp_path))
    assert similarity("", skill) == 0.0


def test_multiple_skills_ranked(tmp_path):
    robot = parse_skill(make_skill(tmp_path, "validate-robot"))
    other = parse_skill(make_skill(tmp_path, "review-code"))

    # A description with no robot-specific vocabulary
    other = type(other)(
        name="review-code",
        description="Review Python formatting and style",
        path=other.path,
        instructions=other.instructions,
    )

    matches = match_skills(
        "Validate robot Q-learning code and tests",
        [other, robot],
    )

    assert matches
    assert matches[0][0].name == "validate-robot"

def test_selected_skill_appears_in_prompt(tmp_path):
    from mini_coding_agent import MiniAgent

    skill = parse_skill(make_skill(tmp_path))

    # Test prompt construction without starting Ollama.
    agent = MiniAgent.__new__(MiniAgent)
    agent.active_skill = skill
    agent.prefix = "You are a coding agent."
    agent.memory_text = lambda: "Memory: empty"
    agent.history_text = lambda: "- empty"

    result = agent.prompt("Validate my robot")

    assert "Selected agent skill instructions" in result
    assert "validate-robot" in result
    assert "Run the robot validation script" in result

