# Intent — Project 2: Agent Skills Component

## Project purpose
Extend the open-source mini coding agent with a seventh component, **Agent Skills**, while retaining its existing coding tools, permission checks, and session behavior.

The component should discover local Markdown skill definitions, identify a skill relevant to a user's prompt, load its instructions into the agent's working context, and allow the agent to request a skill later during execution. It should also record enough information to explain why a skill was chosen and what actions were taken.

## Problem
The starting mini coding agent can inspect files and execute tools, but has no reusable mechanism for discovering and applying task-specific skill instructions. A user or model would otherwise need to provide those instructions manually for every task.

## Intended users
Students demonstrating a skill-augmented coding agent; a teammate who needs to install, run, review, and present the project; and a reviewer who needs to reproduce the tests and Lab 4 demonstration.

## Goals
1. Discover `SKILL.md` or `skill.md` files within a workspace `skills/` directory.
2. Parse each skill's name, description, and Markdown instructions.
3. Compare a user request to discovered skill metadata and select a matching skill when one passes a similarity threshold.
4. Make selected instructions available to the existing model prompt.
5. Support explicit `activate_skill` tool calls while an agent task is in progress.
6. Preserve existing approval checks for shell commands and file modifications.
7. Log prompts, similarity matches, selected skills, and tool-call results in a session-specific JSON file.
8. Demonstrate the existing Lab 5 `validate-robot` skill against the Lab 4 Q-learning vacuum robot files.
9. Keep changes in the group's repository, isolated from the upstream mini-coding-agent repository.

## Scope and boundaries
**In scope:** a separate `agent_skills.py` module, integration with `MiniAgent`, automated tests, logging, the Lab 5 validator and supporting robot files, usage documentation, and a short demonstration.

**Out of scope:** replacing the model backend, rebuilding the Lab 4 robot, retraining Q-learning, modifying its stored Q-table, or contributing changes to the upstream mini-coding-agent repository.

The Lab 5 skill explicitly instructs the agent to validate and report issues, **not** repair or retrain the robot.

## Success criteria
- Skill discovery and selection work under automated tests.
- Irrelevant requests do not select the example skill in the tested case.
- Explicit mid-execution activation makes skill instructions available in the next model prompt.
- Skill decisions and tool results are persisted and survive resumed sessions.
- The original agent tests continue to pass.
- An Ollama-backed run uses the Lab 5 skill to execute `python .opencode/scripts/validate_robot.py` with approval enabled and reports successful checks.

## Evidence recorded during development
- Automated test suite: **34 passed, 1 skipped** at the last reported run.
- Real demonstration model: **`qwen3:1.7b`** using Ollama.
- Logged skill: **`validate-robot`**, match score **0.6666666666666666**.
- Logged tool: `run_shell`, command `python .opencode/scripts/validate_robot.py`.
- Result: exit code **0**, all syntax, movement, and memory validation checks passed.

## Constraints and risks
- Smaller models may not consistently emit valid tool calls; a successful demonstration does not prove reliability across all prompts or hardware.
- Word-overlap similarity can miss semantically relevant paraphrases or match incidental shared terms.
- Skills are local instructions, not independently verified code. Their instructions must not bypass the agent's existing tool approval and validation rules.
- The group repository contains prior coursework and must not be overwritten. The final publication layout is still to be selected and verified; development continues in the original `mini-coding-agent` directory.

## Ownership and handoff
The implementation and initial demonstration were completed in the local `mini-coding-agent` workspace. Before handoff, add Agent Skills instructions to the existing README without replacing its original content; then have the teammate reproduce the workflow and coordinate the presentation. Publish only to the class group repository after verifying the destination and preserving prior work. No push to the upstream repository is part of this project.
