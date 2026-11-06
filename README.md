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

## Running the OpenCode AI Features

The vacuum robot can run independently using Python, but the Lab 5 hooks, skills, and sub-agents require OpenCode.

1. Download and install OpenCode from https://opencode.ai/.
2. Open OpenCode Desktop and select the repository's main project folder.
3. Configure an AI provider and model in OpenCode. An available model is required to run the AI agents.
4. Make sure the .opencode folder is included in the project. This contains the hook, skill, and sub-agent configurations.

To test the three features:

Hooks: Ask OpenCode to review the project. When the agent finishes, the hook automatically runs the Python validator. The results can be viewed in .opencode/hook-results.log.

Skills: Ask OpenCode to load the validate-robot skill using its skill tool and follow the validation procedure. The skill will guide OpenCode through checking the robot and running the Python validator.

Sub-agents: Ask OpenCode to delegate an environment review to environment-reviewer and a Q-learning review to learning-reviewer. Both agents should run in separate sessions and return their findings to the main agent.

The OpenCode features were developed and tested using OpenCode Desktop 2.2.5. The Python robot does not require OpenCode to run, but the additional Lab 5 AI features do.

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