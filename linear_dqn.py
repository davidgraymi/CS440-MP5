import argparse, json, random, time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gym_robot import GYM_DUBINS_CAR_ENV_ID, dist_2d

class LinearDQNAgent:
    def __init__(self, num_features, num_actions, target_weights_update_freq=100, max_buffer_size=50000):
        self.target_weights_update_freq = target_weights_update_freq
        self.num_updates = 0
        self.replay_buffer = []
        self.max_buffer_size = max_buffer_size
        # TODO(IV): initialize the linear Q-function weights
        self.weights = None
        self.target_weights = None
        
    def save_weights(self, filename):
        np.save(filename, self.weights)

    def load_weights(self, filename):
        loaded = np.load(filename, allow_pickle=False)
        if loaded.shape != self.weights.shape:
            raise ValueError(f"Weights have shape {loaded.shape}; expected {self.weights.shape}.")
        if (not np.issubdtype(loaded.dtype, np.number) or not np.isrealobj(loaded)
                or not np.all(np.isfinite(loaded))):
            raise ValueError("Weights must be a finite real numeric array.")
        self.weights = loaded.astype(float, copy=False)
        self.target_weights = self.weights.copy()

    def act(self, features, epsilon=0.1):
        # TODO(IV): implement epsilon-greedy action selection
        return 0

    def update(self, features, action, reward, next_features, terminated, alpha=0.005, gamma=0.9, batch_size=64):
        # TODO(IV): implement the replay-buffer DQN update
        # A standard pattern is:
        # 1. push the new transition into self.replay_buffer
        # 2. once the buffer has at least batch_size transitions, sample a batch
        # 3. compute current Q-values and target Q-values
        # 4. apply a linear gradient step to only the chosen action column
        # 5. periodically copy self.weights into self.target_weights
        pass


def run_episode(
        agent: LinearDQNAgent, env: gym.Env, epsilon,
        learning_rate, discount_factor, batch_size):
    # TODO(IV): run one complete training episode in one environment
    # - reset the environment
    # - select and execute actions until the episode terminates or truncates
    # - pass each transition and the update hyperparameters to agent.update(...)
    # - return the episode's total undiscounted reward
    return 0.0


def train(
        agent: LinearDQNAgent, envs: list[gym.Env],
        num_episodes, epsilon_start, epsilon_end, epsilon_decay_steps,
        learning_rate, discount_factor, batch_size, output_path: Path | None = None):
    # TODO(IV): build your training experiment around run_episode(...)
    # There are many reasonable ways to implement this but 
    # as a baseline you should cycle through every environment and linearly decay epsilon
    return []


def test(agent: LinearDQNAgent, env: gym.Env, output_path: Path | None = None):
    state, _ = env.reset(seed=0)
    total_reward = 0.0
    path = [env.config.copy()]
    done = False
    rewards = []
    terminated = False
    truncated = False

    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        rewards.append(float(reward))
        state = next_state
        path.append(env.config.copy())
        done = terminated or truncated

    path = np.array(path)
    fig, _ = env.render(
        path=path,
        title=f"{env.env_name}: Total Reward {total_reward:.3f}",
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    reached_goal = dist_2d(env.config, env.goal_config) <= env.car_params.goal_tolerance
    return {
        "env_name": env.env_name,
        "total_reward": float(total_reward),
        "num_steps": int(len(path) - 1),
        "step_rewards": rewards,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reached_goal": bool(reached_goal),
        "image_path": str(output_path) if output_path is not None else None,
    }

# helpers for loading and creating environments from the provided JSON files
def numeric_env_sort_key(path: Path):
    return int(path.stem.split("_")[-1])
def make_env(params_file: Path):
    return gym.make(
        GYM_DUBINS_CAR_ENV_ID,
        params_file=str(params_file),
        disable_env_checker=True,
    ).unwrapped


def next_results_dir(base_dir: Path):
    results_root = base_dir / "results"
    run_idx = 1
    while (results_root / f"run_{run_idx}").exists():
        run_idx += 1
    return results_root / f"run_{run_idx}"


def print_test_summary(test_results):
    if not test_results:
        print("Test summary: no test results to report.")
        return

    num_successes = sum(result["reached_goal"] for result in test_results)
    mean_reward = float(np.mean([result["total_reward"] for result in test_results]))
    mean_steps = float(np.mean([result["num_steps"] for result in test_results]))
    num_truncated = sum(result["truncated"] for result in test_results)
    failed_envs = [result["env_name"] for result in test_results if not result["reached_goal"]]

    print(f"Test summary: {num_successes}/{len(test_results)} successful episodes")
    print(f"Mean reward: {mean_reward:.3f}, Mean steps: {mean_steps:.1f}, Truncated episodes: {num_truncated}")
    if failed_envs:
        print("Failed environments:", ", ".join(failed_envs))

if __name__ == "__main__":
    # These defaults are reasonable starting values, not a guaranteed best configuration
    parser = argparse.ArgumentParser(description="CS440 MP5 Reinforcement Learning")
    parser.add_argument(
        "--num_episodes", type=int, default=5000,
        help="Number of outer training episodes; each visits every training environment",
    )
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon value for epsilon-greedy exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.1, help="Final epsilon value for epsilon-greedy exploration")
    parser.add_argument("--epsilon_decay_steps", type=int, default=1000, help="Number of episodes over which to linearly decay epsilon")
    parser.add_argument("--target_weights_update_freq", type=int, default=100, help="Number of updates between target network weight updates")
    parser.add_argument("--max_buffer_size", type=int, default=50000, help="Maximum size of the replay buffer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate for the DQN")
    parser.add_argument("--discount_factor", type=float, default=0.9, help="Discount factor for the DQN")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DQN")
    parser.add_argument("--seed", type=int, default=440, help="Random seed for reproducible training (default: 440)")
    parser.add_argument("--test_only", action="store_true", help="Skip training and only run testing with loaded weights")
    parser.add_argument("--weights_file", type=str, default="linear_dqn_weights.npy", help="Weights file to load for --test_only or --resume_training; you can also point this at a saved run such as results/run_3/linear_dqn_weights.npy")
    parser.add_argument("--resume_training", action="store_true", help="Load weights before training and continue optimizing from them")
    parser.add_argument("--results_dir", type=str, default=None, help="Optional directory for saved plots and test outputs")
    args = parser.parse_args()

    if not 0 <= args.seed <= np.iinfo(np.uint32).max:
        parser.error("--seed must be between 0 and 4294967295")
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    train_param_files = sorted(data_dir.glob("train_env_*.json"), key=numeric_env_sort_key)
    test_param_files = sorted(data_dir.glob("test_env_*.json"), key=numeric_env_sort_key)

    if not test_param_files:
        raise FileNotFoundError(f"No test environments found in {data_dir}")
    if not args.test_only and not train_param_files:
        raise FileNotFoundError(
            f"No training environments found in {data_dir}. Create files named train_env_*.json before training."
        )

    print(f"Loading {len(train_param_files)} training environments and {len(test_param_files)} test environments")
    train_envs = [make_env(params_file) for params_file in train_param_files]
    test_envs = [make_env(params_file) for params_file in test_param_files]
    for env_index, env in enumerate(train_envs):
        env_seed = args.seed + env_index
        env.action_space.seed(env_seed)
        env.reset(seed=env_seed)

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = base_dir / results_dir
    else:
        results_dir = next_results_dir(base_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {results_dir}")

    example_env = train_envs[0] if train_envs else test_envs[0]
    if len(example_env.observation_space.shape) != 1:
        raise ValueError(f"Expected a flat observation vector, got shape {example_env.observation_space.shape}")
    num_features = int(example_env.observation_space.shape[0])
    print(f"Observation space has {num_features} features")
    agent = LinearDQNAgent(
        num_features=num_features,
        num_actions=int(example_env.action_space.n),
        target_weights_update_freq=args.target_weights_update_freq,
        max_buffer_size=args.max_buffer_size,
    )

    weights_path = Path(args.weights_file)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path
    if args.test_only or args.resume_training:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}. "
                "Copy a saved run's linear_dqn_weights.npy into the template directory or pass --weights_file with the run-specific path."
            )
        print(f"Loading weights from {weights_path}")
        agent.load_weights(weights_path)

    if not args.test_only:
        num_episodes = args.num_episodes
        epsilon_decay_steps = args.epsilon_decay_steps if args.epsilon_decay_steps is not None else num_episodes
        start_time = time.time()
        train(
            agent,
            train_envs,
            num_episodes=num_episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            batch_size=args.batch_size,
            output_path=results_dir / "training_returns.png",
        )
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    test_results = []
    for env in test_envs:
        test_results.append(test(agent, env, output_path=results_dir / f"{env.env_name}_path.png"))

    print_test_summary(test_results)

    with open(results_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    if not args.test_only:
        saved_weights_path = results_dir / "linear_dqn_weights.npy"
        agent.save_weights(saved_weights_path)
        print(f"Saved weights to {saved_weights_path}")
        print(
            "To reuse these weights later, either copy this file to "
            f"{base_dir / 'linear_dqn_weights.npy'} or pass --weights_file {saved_weights_path}"
        )
