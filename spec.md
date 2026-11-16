# Specification — Agent Skills Component

## Architecture
The project extends `mini_coding_agent.py` with a separate `agent_skills.py` module.

```text
User request
   | 
   v
MiniAgent.ask() -> discover_skills() -> match_skills()
   |                                  |
   |                                  +-> select skill / none
   v
MiniAgent.prompt() -> load_skill(active_skill) -> model
   |                                              |
   |                          optional activate_skill tool call
   |                                              |
   +------------------------ next prompt <--------+
   |
   v
Existing tool dispatcher -> permission checks -> tool result
   |
   v
Session-specific JSON skill audit log
```

## Skill file format
- Files are recursively discovered under `skills/` when named `SKILL.md` or `skill.md` (case-insensitive matching in the implemented discovery function).
- A skill may contain `---`-delimited, YAML-style metadata with `name` and `description` fields.
- The parser uses straightforward line-based metadata parsing rather than a general YAML library; complex YAML features are not guaranteed.
- If metadata is missing, the parser derives the name from the containing folder and a fallback description from content.
- The full Markdown file is retained as the instructions.

Example:

```markdown
---
name: validate-robot
description: Validate the vacuum robot's movement logic and Q-learning memory.
---

# Instructions
Inspect the robot files and run the project's validation script.
```

## Component interface (`agent_skills.py`)

| Function / type | Role |
|---|---|
| `Skill` | Immutable record of name, description, file path, and instructions |
| `parse_skill(path)` | Parse a Markdown skill file |
| `discover_skills(directory)` | Recursively find and parse matching filenames |
| `tokenize(text)` | Produce normalized lowercase alphanumeric tokens |
| `similarity(prompt, skill)` | Compute word overlap against skill name and description |
| `match_skills(prompt, skills, threshold=0.15)` | Return threshold-passing matches ordered by score |
| `load_skill(skill)` | Format a selected skill for inclusion in model context |

The similarity implementation uses:

`score = count(unique prompt tokens intersecting skill metadata tokens) / count(unique skill metadata tokens)`

This is a **lexical** score, not an embedding/cosine similarity score. A score of 0.6667 is not a 66.67% probability of correctness.

## Agent integration
- During initialization, the agent discovers workspace skills, initializes `active_skill`, and establishes a session-specific log path.
- At the start of `ask(user_message)`, the agent refreshes available skills, calculates matches, and selects the highest-ranked match or no skill.
- `prompt()` includes the selected skill's instructions alongside existing memory, history, and request context.
- `activate_skill` is registered as an agent tool accepting an argument object containing `name`.
- The activation handler looks up a discovered skill by name, changes `active_skill`, and returns an activation result. The next generated prompt includes the new skill instructions.
- Existing `run_tool` validation, repeat-call protection, and approval policy remain in effect. Activating a skill itself is a safe tool operation; subsequent shell or file modification actions still require approval according to the configured policy.

## Logging
Skill audit records are kept in a session-specific JSON file alongside the agent's session storage, using a filename with suffix `_skills.json`.

Each request record includes:

```json
{
  "prompt": "...",
  "matches": [{"skill": "validate-robot", "score": 0.6666666666666666}],
  "selected_skill": "validate-robot",
  "tool_calls": [
    {
      "tool": "run_shell",
      "args": {"command": "python .opencode/scripts/validate_robot.py"},
      "result": "exit_code: 0 ... ALL CHECKS PASSED"
    }
  ]
}
```

If activated through an agent tool, the code also records `activation_source: "agent_tool"`. Records are loaded again when a saved session is resumed.

**Interpretation limit:** a successful selected-skill log and tool result prove the workflow was invoked; a log alone does not establish that every instructional step was followed or that a task was solved independently.

## Lab 4 / Lab 5 demo files

```text
mini-coding-agent/
  agent_skills.py
  mini_coding_agent.py
  pyproject.toml
  skills/validate-robot/SKILL.md
  .opencode/scripts/validate_robot.py
  main.py
  vacuum_env.py
  q_learning_agent.py
  q_memory.json
  tests/test_agent_skills.py
  tests/test_mini_coding_agent.py
```

The skill checks Python syntax, a 4×4 grid and 80/10/10 movement behavior, +10 goal reward, −1 trap reward, and persistence of Q-learning memory. Its validation script uses a temporary memory file for save/load testing; the stored `q_memory.json` should not be overwritten.

## Tests and acceptance
Reported automated result: **34 passed, 1 skipped**. Tests cover skill discovery, matching, prompt inclusion, automatic and explicit activation, log persistence, and preservation across session resumption in addition to existing agent behavior.

Reported end-to-end result: Ollama model `qwen3:1.7b` requested `run_shell` for `python .opencode/scripts/validate_robot.py` under `--approval ask`; after approval, output included `ALL CHECKS PASSED`, and the saved audit log recorded an exit code of 0.

## Known limitations / follow-up checks
- Current lexical matching is basic; evaluate additional paraphrased and ambiguous prompts if time permits.
- Review robustness of metadata parsing for malformed or complex frontmatter.
- Confirm teammate hardware and Ollama availability separately; running OpenCode with a hosted default model does not guarantee local Ollama support.
- Check the final Git-tracked file list and run tests in the original `mini-coding-agent` development workspace. After selecting a safe group-repository publication method, repeat verification on the published copy.
- Verify the exact professor-required documentation/presentation format against the assignment before submitting.
