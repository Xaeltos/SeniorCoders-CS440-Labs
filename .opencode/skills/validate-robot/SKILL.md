
---
name: validate-robot
description: Validate the vacuum robot's movement logic, Python code, and persistent Q-learning memory using the project's automated test script.
---

# Vacuum Robot Validation Skill

## Purpose

Validate the vacuum robot reinforcement-learning project
without modifying its existing code or trained Q-table.

## Project Structure

- main.py: Training and demonstration
- vacuum_env.py: Grid environment and movement logic
- q_learning_agent.py: Q-learning and persistent memory
- q_memory.json: Previously trained Q-table

## Validation Procedure

1. Inspect the existing Python files.
2. Verify the 4x4 grid configuration.
3. Confirm movement follows 80/10/10 probabilities.
4. Verify that the goal reward is +10.
5. Verify that the trap reward is -1.
6. Confirm that Q-table memory can be saved and loaded.
7. Run the existing validation script:

   python .opencode/scripts/validate_robot.py

8. Review the validation output.
9. Report whether each check passed or failed.

## Restrictions

- Do not modify main.py.
- Do not modify vacuum_env.py.
- Do not modify q_learning_agent.py.
- Do not overwrite q_memory.json.
- Do not retrain the Q-learning model.
- Report problems rather than automatically fixing them.

## Expected Result

Provide a summary including:

- Python syntax validation
- Movement validation
- Persistent memory validation
- Overall PASS or FAIL
