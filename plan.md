# Plan — Project 2: Agent Skills

## Development approach
Implement and verify the smallest coherent Agent Skills component, then demonstrate it on the real Lab 4 / Lab 5 artifacts. Maintain a safe separation between the upstream mini-agent repository and the class group's repository.

## Completed milestones

| Milestone | Evidence | Status |
|---|---|---|
| Clone mini-coding-agent and create feature branch | Local `feature/agent-skills` branch | Completed |
| Establish baseline | 18 passed, 1 skipped | Completed |
| Implement skill discovery and matching | `agent_skills.py`, 9 focused tests | Completed |
| Integrate selection into model prompt | Automatic selection and prompt tests | Completed |
| Add mid-execution activation tool | `activate_skill` test | Completed |
| Add selection/tool logging and session resume | Persistent JSON logging tests | Completed |
| Import Lab 5 skill and validation script | `skills/validate-robot/SKILL.md`, `.opencode/scripts/validate_robot.py` | Completed |
| Reuse Lab 4 files | `main.py`, `vacuum_env.py`, `q_learning_agent.py`, `q_memory.json` | Completed |
| Run validator directly | `ALL CHECKS PASSED` | Completed |
| Run actual Ollama-backed agent | `qwen3:1.7b`, tool approval and validator success | Completed |
| Complete regression suite | 34 passed, 1 skipped (reported) | Completed |
| Save local source checkpoint | Commit `7fe2973` on original working clone | Completed |
| Inspect class group repository | Fetched the group branches and reviewed the Lab 5 branch contents | Completed (read-only) |

## Remaining steps, in order

1. **Finish the existing development workspace.** Keep using the original `mini-coding-agent` folder. Keep its original README, and add an Agent Skills section without replacing existing content. Place the three SDLC documents at the project root.
2. **Review the documentation.** Confirm it describes the current code, known limitations, setup commands, and successful Lab 4 / Lab 5 demonstration; avoid claims about untested behavior.
3. **Verify locally.** Run `python -m pytest -q`, `python -m tabnanny mini_coding_agent.py`, `python -m py_compile mini_coding_agent.py`, and `git diff --check`; inspect intended files before committing.
4. **Prepare teammate handoff.** Have the teammate review setup steps and confirm whether their machine can run Ollama. The OpenCode default experience does not establish Ollama hardware compatibility.
5. **Prepare a safe publication plan.** The group repository's Lab 4 and Lab 5 history must be preserved. Choose the final layout with the group before transferring files. Do not push the unrelated upstream clone history to the group repository or push anything to the original upstream repository.
6. **Verify the destination and publish.** Inspect the branch, remotes, tracked file list, and proposed changes before any push. Do not force push or overwrite existing assignments.
7. **Prepare the presentation.** Explain the seventh component, prompt matching, explicit activation, successful validation result, and audit log within the allotted speaking time.

## Suggested 90-second individual presentation allocation
- 0–15 seconds: problem and seventh component.
- 15–40 seconds: skill discovery/matching and activation.
- 40–65 seconds: Lab 5 skill + Lab 4 validator demonstration.
- 65–80 seconds: audit log and test result.
- 80–90 seconds: limitations and conclusion.

If the 90-second allowance is shared with a teammate, reduce the individual material proportionally.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Accidental upstream push | Keep upstream push disabled; inspect the destination and history before any group-repo push |
| Overwriting prior coursework | Choose and verify a group-approved layout before publication; preserve all existing branches |
| Teammate cannot run local model | Verify Ollama availability/RAM, use small `qwen3:1.7b`, coordinate demo ownership |
| Skill matches incorrectly | Include both positive and negative tests; document lexical-matching limit |
| Agent runs unsafe commands | Demonstrate using `--approval ask`; inspect each proposed shell command |
| Missing handoff context | Add verified Agent Skills documentation to the original README before handoff |

## Definition of done
- Verified group publication includes source, example skill, validator, Lab 4 artifacts, three SDLC files, and an additive update to the original README.
- Tests run successfully in development and on the final group-published copy.
- Demo outcome and audit log can be reproduced.
- A teammate understands how to run and explain the work.
- Presentation and submission comply with the instructor's requirements.
