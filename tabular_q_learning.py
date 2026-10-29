import argparse
from pathlib import Path
import random

import gymnasium as gym # our interface to environments, in particular Taxi-v4
import matplotlib.pyplot as plt
import numpy as np

class QAgent:
    def __init__(self, state_space : gym.Space, action_space : gym.Space, init_val=0.0):
        self._state_space = state_space
        self._action_space = action_space
        # TODO(III): initialize table for Q-values as self.q_table
        self.q_table = None

    def save_weights(self, filename="QAgent_weights.npy"):
        np.save(filename, self.q_table)

    # TODO(III): implement epsilon-greedy action selection
    # return the action index (integer) chosen by the agent for the given state
    def act(self, state, epsilon=0.0):
        return None
    
    # TODO(III): implement the Q-learning update rule to update self.q_table
    # terminal is a boolean indicating whether next_state is a terminal state, if so then the target should be just the reward
    # alpha is the learning rate and gamma is the discount factor
    def update(self, state, action, reward, next_state, terminal, alpha, gamma):
        pass

# TODO(III): implement the training loop for QAgent using a Gymnasium environment
# epochs: number of episodes to train for, an episode ends when the environment step returns either terminated or truncated as True
# alpha: learning rate for Q-learning update
# gamma: discount factor for Q-learning update
# epsilon: exploration rate for epsilon-greedy action selection
# Return a list of episode rewards obtained during training for debugging purposes
def train_agent(env : gym.Env, agent : QAgent,
                epochs=10000, alpha=0.1, 
                gamma=0.9, epsilon=0.1):
    return []

# you may want to consider implementing a separate "evaluation" function
# which checks the performance of the trained agent, e.g., using epsilon=0.0
# The greedy policy extracted from your self.q_table is evaluated by the autograder
def eval_agent(env : gym.Env, agent : QAgent):
    pass

def q_learning(
        epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1,
        init_val=0, save_weights=False, seed=440):
    # Seed all three random-number sources used by this runner. Calling reset
    # with a seed once initializes Gymnasium's generator; later unseeded resets
    # advance that generator instead of repeating the same episode
    random.seed(seed)
    np.random.seed(seed)
    # Create the environment, set the render_mode
    # This render mode means you can print states to debug; see the Taxi-v4 documentation for details
    env = gym.make("Taxi-v4", render_mode="ansi")
    env.action_space.seed(seed)
    env.reset(seed=seed)
    # Initialize our agent
    agent = QAgent(env.observation_space, env.action_space, init_val=init_val)
    # Train
    episode_rewards = train_agent(env, agent, epochs=epochs, alpha=alpha, gamma=gamma, epsilon=epsilon)
    # Evaluation
    eval_agent(env, agent)
    # Save weights
    if save_weights:
        # Submit this final saved weights file to Gradescope for grading.
        weights_path = Path(__file__).resolve().parent / "QAgent_weights.npy"
        agent.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")
    env.close()
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS440 MP5 tabular Q-learning experiments")
    parser.add_argument(
        "--seed", type=int, default=440,
        help="Random seed for Python, NumPy, and the Taxi environment (default: 440)",
    )
    args = parser.parse_args()

    experiments = {
        "pessimistic_init": {
            "epochs": 10, # is this enough?
            "alpha": 0.001, # is this too big? too small?
            # in lecture we assumed gamma was part of the problem description, here you can tune it...
            "gamma": 0.999, # is this too big? too small?
            "epsilon": 0.1, # what is a good exploration rate for this problem? 
            "init_val": -100000000.0, # very pessimistic initialization...
            "save_weights": True 
        }
    }
    # example code for running experiments and plotting the resulting returns
    # NOTE: this code will only work if you return a list of episode rewards from your train_agent function...
    moving_average_window = 10
    moving_average_episode_rewards = {}
    for exp in experiments:
        experiment_params = {**experiments[exp], "seed": args.seed}
        print(f"Running experiment {exp} with parameters: {experiment_params}")
        episode_rewards = q_learning(**experiment_params)
        # create moving average plot
        if len(episode_rewards) >= moving_average_window:
            moving_average_episode_rewards[exp] = np.convolve(np.array(episode_rewards), np.ones(moving_average_window) / moving_average_window, mode='valid')
        
    for exp, mov_avg_ep_rewards in moving_average_episode_rewards.items():
        plt.plot(mov_avg_ep_rewards, label=exp)

    plt.xlabel("Epochs")
    plt.ylabel("Smoothed Episode Returns")
    plt.legend()
    plt.show()
