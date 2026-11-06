import random
import json
from pathlib import Path

class QLearningAgent:
    """
    Q-Learning Agent with Epsilon-Greedy exploration.
    """
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.actions = actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}       # Format: {(state, action): q_value}

    def get_q_value(self, state, action):
        """Returns the Q-value for a state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Selects an action using the Epsilon-Greedy strategy."""
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(self.actions)
        else:
            # Exploit: choose best action based on Q-table
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # Handle ties randomly to prevent bias
            best_actions = [a for a in self.actions if self.get_q_value(state, a) == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, done):
        """Applies the Bellman Equation to update the Q-value."""
        current_q = self.get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            # Find max Q-value for the next state
            next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
            target_q = reward + self.gamma * next_max_q

        # Q-Learning Update Rule
        self.q_table[(state, action)] = current_q + self.alpha * (target_q - current_q)

    def decay_epsilon(self, min_epsilon=0.05, decay_rate=0.995):
        """Reduces epsilon over time to shift from exploration to exploitation."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    
    def save_model(self, filename="q_memory.json"):
        """Save the trained Q-table to a JSON file."""

        data = {
            "q_table": [
                {
                    "state": list(state),
                    "action": action,
                    "value": value
                }
                for (state, action), value in self.q_table.items()
            ]
        }

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Model saved to {filename}")

    def load_model(self, filename="q_memory.json"):
        """Load a previously trained Q-table."""

        path = Path(filename)

        if not path.exists():
            print("No saved memory found.")
            return False

        with open(path, "r") as file:
            data = json.load(file)

        self.q_table = {
            (tuple(item["state"]), item["action"]): item["value"]
            for item in data["q_table"]
        }

        print(f"Model loaded from {filename}")
        return True
