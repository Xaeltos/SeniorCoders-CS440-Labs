from vacuum_env import VacuumEnvironment
from q_learning_agent import QLearningAgent
import time

def train_and_test():
    # The exact grid from the image
    grid_data = [
        ["start", 0, 0, 0],
        [0, "X", 0, 0],
        [0, 0, 0, 0],
        [0, -1, 0, 10]
    ]
    
    # Initialize environment with the grid
    env = VacuumEnvironment(grid=grid_data, living_cost=-0.1)
    agent = QLearningAgent(actions=env.action_space)

    #check for previous memory
    model_loaded = agent.load_model()
    
    
    # Train only if there is no saved Q-table
    if not model_loaded:
        episodes = 1000
        print("Starting Training...")

        for episode in range(episodes):
            state = env.reset()
            done = False
            steps = 0

            while not done and steps < 100:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)

                agent.update_q_table(
                    state, action, reward, next_state, done
                )

                state = next_state
                steps += 1

            agent.decay_epsilon()

            if (episode + 1) % 200 == 0:
                print(
                    f"Episode {episode + 1}/{episodes} completed. "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

        # Save the Q-table after all episodes
        agent.save_model()

    else:
        print("Saved memory found. Skipping training.")

    print("\n--- Training Complete. Starting Test Run ---")
    # Test run with NO exploration
    agent.epsilon = 0.0 
    state = env.reset()
    path = [state]
    done = False
    total_reward = 0
    
    print(f"Initial State: {state}")
    step_num = 1
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state)
        print(f"Step {step_num}: State {state} -> Action {action} -> New State {next_state} (Reward: {reward})")
        state = next_state
        step_num += 1
        time.sleep(0.3) # Pause for visual effect during video recording

    print("\n--- Final Results ---")
    print(f"Total Path: {path}")
    print(f"Total Reward: {total_reward:.2f}")
    
    if state == env.goal_pos:
        print("SUCCESS: The vacuum robot reached the recharge station!")
    elif state == env.trap_pos:
        print("FAILURE: The vacuum robot fell into the trap!")

if __name__ == "__main__":
    train_and_test()