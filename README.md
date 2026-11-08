# Vacuum Robot Q-Learning: Agentic Code Features

## Strategy Overview

This repository builds upon the foundational Q-Learning vacuum robot environment by integrating **Agentic AI features** (Hooks, Skills, and Sub-agents). Instead of just using an AI to write the code, this project demonstrates how to configure the AI's environment to automate testing, enforce constraints, and isolate context.

By utilizing these features, we created a robust, self-validating development loop that ensures the core Q-learning implementation and persistent memory remain unbroken.

## Project Structure

### Core Q-Learning Implementation
- `vacuum_env.py`: The simulated grid environment. It accepts a 2D array, parses the coordinates for Start, Goal, Trap, and Obstacles, and implements the stochastic 80/10/10 transition logic.
- `q_learning_agent.py`: The Q-Learning implementation. Contains the Q-table, epsilon-greedy action selection, and the Bellman equation update rule.
- `main.py`: The training loop (1000 episodes) and the final test script.
- `q_memory.json`: Stores the trained Q-table so the robot can reuse learned values between executions without retraining.

### Agentic Features (`.opencode/`)
- `.opencode/plugins/robot-hooks.js`: An OpenCode plugin (Hook) that automatically triggers validation checks when an agent execution finishes.
- `.opencode/scripts/validate_robot.py`: The automated test script that checks Python syntax, robot movement, and persistent memory.
- `.opencode/skills/validate-robot/SKILL.md`: A reusable AI Skill containing instructions on how to validate the robot.
- `.opencode/agents/environment-reviewer.md`: A read-only Sub-agent responsible for reviewing the grid environment.
- `.opencode/agents/learning-reviewer.md`: A read-only Sub-agent responsible for reviewing Q-learning logic and model persistence.

## Hyperparameter Decisions

The Q-Learning agent uses the following hyperparameters (defined in `q_learning_agent.py` and `main.py`):
- **Living Reward (Step Penalty): `-0.1`** — A small negative penalty encourages the agent to find the shortest path to the goal rather than wandering aimlessly.
- **Discount Factor (Gamma): `0.95`** — A high discount factor ensures the agent values future rewards (reaching the +10 goal) highly, which is crucial for navigating a grid where the goal is several steps away.
- **Learning Rate (Alpha): `0.1`** — A moderate learning rate that allows the agent to steadily update its Q-values without overwriting previous knowledge too aggressively.
- **Epsilon (Exploration Rate): `1.0` (decaying to `0.05`)** — The agent starts by exploring entirely randomly (1.0) and decays its exploration rate by 0.995 per episode, eventually relying on its learned Q-table (exploitation).

## How to Run

1. Ensure you have Python 3 installed.
2. Run the main script:
   ```
   python main.py
   ```
3. When the program starts, it checks for q_memory.json.
   * If saved memory exists, the robot loads its trained Q-table and skips training.
   * Otherwise, it trains for 1000 episodes, saves the model, and then runs the final demonstration.

## Set Up OpenCode and Reproduce the Lab 5 Features

The vacuum robot runs independently with Python. **OpenCode is only required to demonstrate the Lab 5 hook, skill, and sub-agents.** The project was developed and tested with **OpenCode Desktop 2.2.5**; UI details may differ in other versions.

### 1. Get the repository and prerequisites

1. Download or clone this complete repository to a local folder. Keep the `.opencode` directory and its contents: folders whose names start with a dot can be hidden by some file managers and archive tools.
2. Install **Python 3** and confirm that `python --version` works in a terminal/PowerShell window. On Windows, `py --version` may work instead.
3. From the repository root (the directory containing `main.py`), run the robot first:

   ```powershell
   python main.py
   ```

   If your Windows system uses the Python launcher instead, run `py main.py`. The program loads `q_memory.json` if present or trains and saves a Q-table if not. The validation script is also located at `.opencode/scripts/validate_robot.py`.

### 2. Install and configure OpenCode Desktop

1. Download **OpenCode Desktop** for your operating system from https://opencode.ai/docs/ (choose the **Desktop** download). Install and launch it.
2. Open the **repository root folder** in OpenCode — **not** only the `.opencode` folder or a single `.py` file. The root should contain `main.py`, `vacuum_env.py`, `q_learning_agent.py`, and `.opencode/`.
3. In OpenCode, connect a supported AI provider/account and select a model available to that provider. Follow OpenCode's provider connection prompts. **An available model and any required provider authentication are necessary** to run the AI review tasks; this project's Python robot itself does not require an AI provider.
4. Start a **new OpenCode conversation/session** after opening the repository, so project-local agents, skills, and plugins can be discovered. If you added or changed `.opencode` files while OpenCode was open, reopen/reload the project.

You **do not need to copy the `.opencode` files into your global OpenCode configuration**. OpenCode discovers the following project-local locations:

```text
project-root/
├── main.py
├── vacuum_env.py
├── q_learning_agent.py
├── q_memory.json                  (created/loaded by robot)
└── .opencode/
    ├── plugins/
    │   └── robot-hooks.js
    ├── scripts/
    │   └── validate_robot.py
    ├── skills/
    │   └── validate-robot/
    │       └── SKILL.md
    └── agents/
        ├── environment-reviewer.md
        └── learning-reviewer.md
```

### 3. Demonstrate the hook

The project-local plugin `.opencode/plugins/robot-hooks.js` is intended to run the Python validator after the configured agent-completion event.

1. With the repository open, start a main-agent conversation in OpenCode.
2. Enter this prompt:

   > Review the vacuum robot project and summarize its main components. Do not modify project files.

3. Let the agent finish its task, then check `.opencode/hook-results.log` in the repository. The hook is expected to invoke `.opencode/scripts/validate_robot.py`, and the log should contain its validation results.
4. If no log appears, ensure OpenCode is opened at the project root, the plugin file exists, and Python is available to the OpenCode process. Consult the plugin source for its exact trigger and error-handling behavior.

### 4. Demonstrate the reusable skill

OpenCode discovers skills located under `.opencode/skills/`; the skill's full instructions are loaded on demand.

In the main-agent conversation, use this prompt:

> Load the `validate-robot` skill using your skill tool, follow its instructions, and report the robot validation results. Do not edit the robot source code.

Check that OpenCode actually **loads** the `validate-robot` skill and performs the requested checks. The skill definition is `.opencode/skills/validate-robot/SKILL.md`, and it references the validator in `.opencode/scripts/validate_robot.py`.

### 5. Demonstrate the two read-only sub-agents

Project-local custom agents are defined in `.opencode/agents/`. The two reviewers are intended to analyze separate concerns without editing files.

In the main-agent conversation, enter:

> Delegate a review of the grid environment, transition probabilities, obstacles, and rewards to `environment-reviewer`. Separately delegate a review of Q-learning updates, action selection, and Q-table persistence to `learning-reviewer`. Use separate sub-agent sessions and return both findings. Do not modify project files.

Confirm that OpenCode invokes **both named sub-agents**, that each returns findings, and that source files remain unchanged. Their read-only permissions are described in their respective `.md` files.

### 6. Troubleshooting and reproducibility notes

- **OpenCode can't find the skill or reviewers:** Check the directory spelling exactly and make sure OpenCode opened the repository root. Restart or reopen the project after changes.
- **The hook does not run:** Check `.opencode/plugins/robot-hooks.js`, whether the completion event occurred, and whether Python can be launched. A manual run of `python .opencode/scripts/validate_robot.py` can help distinguish a Python validation problem from a hook-loading problem; **a manual run alone is not proof the hook fired**.
- **The agent will not start:** Select an accessible model and complete the provider's authentication/configuration. This is separate from running `python main.py`.
- **Hidden project folder missing:** Ensure `.opencode/` was included when cloning or extracting the repository.
- **Q-table differs between machines:** When `q_memory.json` is absent, the robot may generate and save new learned values rather than loading an existing table.

OpenCode documentation: https://opencode.ai/docs/ · [Skills](https://opencode.ai/docs/skills) · [Agents](https://opencode.ai/docs/agents/) · [Plugins](https://opencode.ai/docs/plugins/).

## Lab 5 - Agentic AI Features

For Lab 5, I expanded the vacuum robot project using three features in OpenCode: Hooks, Skills, and Sub-agents. These features help automate testing and code reviews without changing the original Q-learning implementation.

1. Hooks
I created a hook in .opencode/plugins/robot-hooks.js that runs automatically after OpenCode finishes executing a task. It triggers the Python validation script in .opencode/scripts/validate_robot.py, which checks the files for syntax errors, tests the robot's movement behavior, and verifies that the Q-table can be saved and loaded.
The results are written to .opencode/hook-results.log. I tested the hook and confirmed that it ran automatically and passed all checks.

2. Skills
I created a reusable skill in .opencode/skills/validate-robot/SKILL.md. The skill gives OpenCode strict instructions for checking the grid environment, movement probabilities, rewards, and saved Q-table.
I tested it by having OpenCode load the skill and follow its validation instructions. The skill loaded correctly, and all tests passed.

3. Sub-agents
I created two sub-agents to review different parts of the project, demonstrating context isolation:
   * The environment-reviewer checks the grid, obstacles, movement probabilities, and rewards.
   * The learning-reviewer checks the Q-learning calculations, action selection, and persistent memory.

Both agents have read-only permissions (effect: deny) so they cannot modify the project. I tested them by having OpenCode launch both in separate sessions. They completed their reviews and reported their findings back to the main agent.

## Results
All three features were successfully tested. The hook automatically ran validation, the skill loaded and executed correctly, and both sub-agents completed their reviews. The original Python code and trained Q-table were left unchanged.