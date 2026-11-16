&nbsp;
# Mini-Coding-Agent

This folder contains a small standalone coding agent:

- code: `mini_coding_agent.py`
- CLI: `mini-coding-agent`

It is a minimal local agent loop with:

- workspace snapshot collection
- stable prompt plus turn state
- structured tools
- approval handling for risky tools
- transcript and memory persistence
- bounded delegation

The model backend is currently based on Ollama.

<a href="https://magazine.sebastianraschka.com/p/components-of-a-coding-agent">
  <img src="https://substack-post-media.s3.amazonaws.com/public/images/49b97718-57f4-4977-99c8-8ad5c4d32af3_1548x862.png" width="500px">
</a>

<br>

**[The detailed tutorial: Components of a Coding Agent](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)**


&nbsp;
## Six Core Components

<a href="https://magazine.sebastianraschka.com/p/components-of-a-coding-agent">
  <img alt="Six core components of a coding agent" src="https://sebastianraschka.com/images/github/mini-coding-agent/six-components.webp" width="500px">
</a>

This coding harness is organized around six practical building blocks:

1. **Live repo context**  
   The agent collects stable workspace facts upfront, such as repo layout, instructions, and git state.
2. **Prompt shape and cache reuse**  
   A stable prompt prefix, which is separate from the changing request, transcript, and memory so repeated model calls can reuse the static parts efficiently.
3. **Structured tools, validation, and permissions**  
   The model works through named tools with checked inputs, workspace path validation, and approval gates instead of free-form arbitrary actions.
4. **Context reduction and output management**  
   Long outputs are clipped, repeated reads are deduplicated, and older transcript entries are compressed to keep prompt size under control.
5. **Transcripts, memory, and resumption**  
   The runtime keeps both a full durable transcript and a smaller working memory so sessions can be resumed while preserving important state via working memory.
6. **Delegation and bounded subagents**  
   Scoped subtasks can be delegated to helper agents that inherit enough context to help (but operate within limits).

&nbsp;
## Requirements

You need:

- Python 3.10+
- Ollama installed
- an Ollama model pulled locally

Optional:

- `uv` for environment management and the `mini-coding-agent` CLI entry point

This project has no Python runtime dependency beyond the standard library, so you can run it directly with `python mini_coding_agent.py` if you do not want to use `uv`.

&nbsp;
## Install Ollama

Install Ollama on your machine so the `ollama` command is available in your shell.

Official installation link: [ollama.com/download](https://ollama.com/download)

Then verify:

```bash
ollama --help
```

Start the server:

```bash
ollama serve
```

In another terminal, pull a model. Example:

```bash
ollama pull qwen3.5:4b
```

Qwen 3.5 model library:

- [ollama.com/library/qwen3.5](https://ollama.com/library/qwen3.5)

The default in this project is `qwen3.5:4b`. If you have sufficient memory, it is worth trying a larger model such as `qwen3.5:9b` or another larger Qwen 3.5 variant. The agent just sends prompts to Ollama's `/api/generate` endpoint.

&nbsp;
## Project Setup

Clone the repo or your fork and change into it:

```bash
git clone https://github.com/rasbt/mini-coding-agent.git
cd mini-coding-agent
```

If you forked it first, use your fork URL instead:

```bash
git clone https://github.com/<your-github-user>/mini-coding-agent.git
cd mini-coding-agent
```



&nbsp;
## Basic Usage

Start the agent:

```bash
cd mini-coding-agent
uv run mini-coding-agent
```

Without `uv`, run the script directly:

```bash
cd mini-coding-agent
python mini_coding_agent.py
```

By default it uses:

- model: `qwen3.5:4b`
- approval: `ask`

For a concrete usage example, see [EXAMPLE.md](EXAMPLE.md).

&nbsp;
## Approval Modes

Risky tools such as shell commands and file writes are gated by approval.

- `--approval ask`
  prompts before risky actions (default and recommended)
- `--approval auto`
  allows risky actions automatically, including arbitrary command execution and file writes by the model; use only with trusted prompts and trusted repositories
- `--approval never`
  denies risky actions

Example:

```bash
uv run mini-coding-agent --approval auto
```



&nbsp;
## Resume Sessions

The agent saves sessions under the target workspace root in:

```text
.mini-coding-agent/sessions/
```

Resume the latest session:

```bash
uv run mini-coding-agent --resume latest
```


Resume a specific session:

```bash
uv run mini-coding-agent --resume 20260401-144025-2dd0aa
```


&nbsp;
## Interactive Commands

Inside the REPL, slash commands are handled directly by the agent instead of
being sent to the model as a normal task.

- `/help`
  shows the list of available interactive commands
- `/memory`
  prints the distilled session memory, including the current task, tracked files, and notes
- `/session`
  prints the path to the current saved session JSON file
- `/reset`
  clears the current session history and distilled memory but keeps you in the REPL
- `/exit`
  exits the interactive session
- `/quit`
  exits the interactive session; alias for `/exit`

&nbsp;
## Main CLI Flags

```bash
uv run mini-coding-agent --help
```

Without `uv`:

```bash
python mini_coding_agent.py --help
```

CLI flags are passed before the agent starts. Use them to choose the workspace,
model connection, resume behavior, approval mode, and generation limits.

Important flags:

- `--cwd`
  sets the workspace directory the agent should inspect and modify; default: `.`
- `--model`
  selects the Ollama model name, such as `qwen3.5:4b`; default: `qwen3.5:4b`
- `--host`
  points the agent at the Ollama server URL (usually not needed); default: `http://127.0.0.1:11434`
- `--ollama-timeout`
  controls how long the client waits for an Ollama response (usually not needed); default: `300` seconds
- `--resume`
  resumes a saved session by id or uses `latest`; default: start a new session
- `--approval`
  controls how risky tools are handled: `ask`, `auto`, or `never`; default: `ask`
- `--max-steps`
  limits how many model and tool turns are allowed for one user request; default: `6`
- `--max-new-tokens`
  caps the model output length for each step; default: `512`
- `--temperature`
  controls sampling randomness; default: `0.2`
- `--top-p`
  controls nucleus sampling for generation; default: `0.9`

&nbsp;
## Example

See [EXAMPLE.md](EXAMPLE.md)

&nbsp;
## Notes & Tips

- The agent expects the model to emit either `<tool>...</tool>` or `<final>...</final>`.
- Different Ollama models will follow those instructions with different reliability.
- If the model does not follow the format well, use a stronger instruction-following model.
- The agent is intentionally small and optimized for readability, not robustness.

---

## Component 7: Agent Skills (CS440 Project 2)

This project extends the original Mini-Coding-Agent with a seventh component: **Agent Skills**.

The component allows the agent to discover reusable skills, match them to user requests, activate them during execution, and record which skills and tools were used.

The original six components and their permission controls remain intact.

### Architecture

The Agent Skills component is implemented in `agent_skills.py` and integrated with `mini_coding_agent.py`.

Its main capabilities include:

1. **Skill Discovery:** Recursively searches the workspace's `skills/` directory for Markdown files named `SKILL.md` or `skill.md`.
2. **Metadata Parsing:** Extracts skill names, descriptions, and instructions.
3. **Similarity Matching:** Compares user requests against skill metadata using normalized word overlap.
4. **Automatic Activation:** Selects the highest-scoring matching skill when its score meets the configured threshold.
5. **Explicit Activation:** Provides an `activate_skill` tool so the agent can activate a skill during execution.
6. **Instruction Loading:** Adds selected skill instructions to the model's prompt.
7. **Execution Logging:** Records prompts, matching scores, selected skills, and tool calls in persistent JSON files.

The default similarity threshold is `0.15`. Similarity scores measure lexical word overlap, not the probability that a skill is correct.

### Example Skill: validate-robot

The example skill is located at:

`skills/validate-robot/SKILL.md`

This skill was originally developed during Lab 5 and is reused to validate the Lab 4 Q-learning vacuum robot.

The validation script is located at:

`.opencode/scripts/validate_robot.py`

The robot implementation includes:

- `main.py`
- `vacuum_env.py`
- `q_learning_agent.py`
- `q_memory.json`

The skill instructs the agent to inspect the robot implementation, check movement behavior and memory persistence, execute the validation script, and report the results.

The skill explicitly prohibits modifying the robot implementation, overwriting its trained Q-table, or retraining the model.

### Running the Lab 4 Demonstration

Requirements:

- Python 3.10 or newer
- Ollama installed and running
- A locally available Ollama model

The successful demonstration used `qwen3:1.7b`, a lightweight alternative to the upstream agent's default model.

Download it with:

```powershell
ollama pull qwen3:1.7b
```

From the project root, execute:

```powershell
python mini_coding_agent.py --model qwen3:1.7b --approval ask --max-steps 10 "Use the validate-robot skill to inspect and validate the Lab 4 vacuum robot project. Follow the skill instructions, run its validation script, and report the results. Do not modify or retrain the robot."
```

When prompted to approve the shell command, verify that the requested command is:

```powershell
python .opencode/scripts/validate_robot.py
```

Approve this command to allow the validation script to execute.

The `--approval ask` option preserves user control over potentially risky tool operations.

### Demonstration Results

The recorded end-to-end test produced:

| Measurement | Result |
|---|---|
| Model | Qwen3 1.7B |
| Selected skill | validate-robot |
| Similarity score | 0.6667 |
| Executed tool | run_shell |
| Command exit code | 0 |
| Python syntax validation | PASS |
| Movement validation | PASS |
| Persistent memory validation | PASS |
| Overall validation | ALL CHECKS PASSED |

The audit log confirmed that the agent selected the skill, requested execution of the correct validation script, and recorded its successful result.

This demonstrates successful skill-guided validation of an existing Lab 4 implementation, not independent reconstruction of the robot project.

### Skill Execution Logs

The component stores session-specific audit logs under:

`.mini-coding-agent/sessions/`

Each skill log uses the filename format:

`<session-id>_skills.json`

Records contain:

- Original user prompt
- Matching skills and similarity scores
- Selected skill
- Tool calls, arguments, and returned results
- Explicit activation source when applicable

The component also reloads previous skill audit records when an existing session resumes.

### Automated Testing

Run the complete test suite from the project root:

```powershell
python -m pytest -q
```

The last reported result was:

**34 passed, 1 skipped**

Tests cover skill discovery, parsing, similarity matching, prompt integration, automatic selection, explicit mid-execution activation, logging, session resumption, and existing Mini-Coding-Agent functionality.

The automated tests use simulated model responses where appropriate. The Lab 4 validation demonstration was also executed using a real Ollama model.

### SDLC Documentation

The project includes three supporting documents:

- `intend.md` — Project purpose, scope, goals, and success criteria.
- `spec.md` — Component architecture, interfaces, behavior, and validation requirements.
- `plan.md` — Development milestones, testing, remaining work, and submission preparation.

### Known Limitations

- Similarity matching uses basic lexical overlap rather than semantic embeddings.
- Skill descriptions with different wording from the user request may not match reliably.
- Small language models may produce inconsistent tool-call formatting.
- The successful demonstration does not guarantee identical behavior across hardware configurations or model versions.
- The agent loads skill instructions but does not independently verify that every instruction was followed.
- Skill activation does not bypass the existing tool validation and permission system.

### Attribution

The original Mini-Coding-Agent was developed by Sebastian Raschka:

https://github.com/rasbt/mini-coding-agent

The Agent Skills extension was implemented as part of CS440 Project 2.
