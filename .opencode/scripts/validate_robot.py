
import ast
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Find the folder containing main.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vacuum_env import VacuumEnvironment
from q_learning_agent import QLearningAgent


GRID = [
    ["start", 0, 0, 0],
    [0, "X", 0, 0],
    [0, 0, 0, 0],
    [0, -1, 0, 10]
]


def check_syntax():
    for filename in [
        "main.py",
        "vacuum_env.py",
        "q_learning_agent.py"
    ]:
        path = ROOT / filename
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=filename)
        print(f"PASS: {filename} syntax")


def check_movement():
    env = VacuumEnvironment(GRID)

    # Test intended Right movement
    env.state = (2, 2)
    with patch("random.random", return_value=0.5):
        state, _, _ = env.step(3)
    assert state == (2, 3)

    # Test first perpendicular slip (Up)
    env.state = (2, 2)
    with patch("random.random", return_value=0.85):
        state, _, _ = env.step(3)
    assert state == (1, 2)

    # Test second perpendicular slip (Down)
    env.state = (2, 2)
    with patch("random.random", return_value=0.95):
        state, _, _ = env.step(3)
    assert state == (3, 2)

    print("PASS: Movement validation")


def check_memory():
    agent = QLearningAgent(actions=[0, 1, 2, 3])
    agent.q_table[((0, 0), 3)] = 5.0

    # Temporary file protects the actual trained memory
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "test_memory.json")

        agent.save_model(path)

        restored = QLearningAgent(actions=[0, 1, 2, 3])
        assert restored.load_model(path)
        assert restored.q_table == agent.q_table

    print("PASS: Persistent memory validation")


if __name__ == "__main__":
    try:
        check_syntax()
        check_movement()
        check_memory()
        print("ALL CHECKS PASSED")
    except Exception as error:
        print(f"VALIDATION FAILED: {error}")
        sys.exit(1)

