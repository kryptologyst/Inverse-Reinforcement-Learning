"""Training script for Maximum Entropy IRL."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
import gymnasium as gym
import torch

from src.algorithms.max_entropy_irl import MaximumEntropyIRL, IRLConfig
from src.envs.data_collection import DataCollector, ExpertPolicy, RandomPolicy
from src.utils.seeding import set_seed


def create_config() -> IRLConfig:
    """Create default IRL configuration.
    
    Returns:
        IRL configuration.
    """
    return IRLConfig(
        hidden_size=64,
        num_layers=2,
        learning_rate=1e-3,
        num_iterations=1000,
        batch_size=32,
        regularization_weight=0.01,
        entropy_weight=1.0,
        eval_frequency=100,
        num_eval_episodes=10
    )


def collect_expert_data(
    env: gym.Env,
    num_trajectories: int = 50,
    expert_type: str = "heuristic",
    seed: int = 42
) -> list:
    """Collect expert demonstration data.
    
    Args:
        env: Gymnasium environment.
        num_trajectories: Number of expert trajectories to collect.
        expert_type: Type of expert policy.
        seed: Random seed.
        
    Returns:
        List of expert trajectories.
    """
    print(f"Collecting {num_trajectories} expert trajectories using {expert_type} policy...")
    
    # Create expert policy
    expert_policy = ExpertPolicy(env, policy_type=expert_type)
    
    # Create data collector
    collector = DataCollector(env, seed=seed)
    
    # Collect trajectories
    expert_trajectories = collector.collect_trajectories(
        expert_policy,
        num_trajectories=num_trajectories,
        max_steps=500
    )
    
    print(f"Collected {len(expert_trajectories)} expert trajectories")
    print(f"Average trajectory length: {np.mean([len(traj) for traj in expert_trajectories]):.2f}")
    
    return expert_trajectories


def collect_policy_data(
    env: gym.Env,
    num_trajectories: int = 50,
    seed: int = 42
) -> list:
    """Collect policy data for comparison.
    
    Args:
        env: Gymnasium environment.
        num_trajectories: Number of policy trajectories to collect.
        seed: Random seed.
        
    Returns:
        List of policy trajectories.
    """
    print(f"Collecting {num_trajectories} policy trajectories using random policy...")
    
    # Create random policy
    random_policy = RandomPolicy(env)
    
    # Create data collector
    collector = DataCollector(env, seed=seed + 1)  # Different seed
    
    # Collect trajectories
    policy_trajectories = collector.collect_trajectories(
        random_policy,
        num_trajectories=num_trajectories,
        max_steps=500
    )
    
    print(f"Collected {len(policy_trajectories)} policy trajectories")
    print(f"Average trajectory length: {np.mean([len(traj) for traj in policy_trajectories]):.2f}")
    
    return policy_trajectories


def train_irl_model(
    env: gym.Env,
    expert_trajectories: list,
    policy_trajectories: list,
    config: IRLConfig
) -> MaximumEntropyIRL:
    """Train the IRL model.
    
    Args:
        env: Gymnasium environment.
        expert_trajectories: Expert demonstration trajectories.
        policy_trajectories: Policy trajectories for comparison.
        config: IRL configuration.
        
    Returns:
        Trained IRL model.
    """
    print("Training Maximum Entropy IRL model...")
    
    # Get state dimension
    state_dim = env.observation_space.shape[0]
    
    # Initialize IRL model
    irl_model = MaximumEntropyIRL(state_dim, config)
    
    # Convert trajectories to tuples
    expert_tuples = [traj.to_tuples() for traj in expert_trajectories]
    policy_tuples = [traj.to_tuples() for traj in policy_trajectories]
    
    # Train the model
    training_history = irl_model.train(expert_tuples, policy_tuples)
    
    print("Training completed!")
    print(f"Final training loss: {training_history['training_losses'][-1]:.4f}")
    print(f"Final eval reward: {training_history['eval_rewards'][-1]:.4f}")
    
    return irl_model


def plot_training_results(
    training_history: Dict[str, list],
    save_path: str
) -> None:
    """Plot training results.
    
    Args:
        training_history: Training history dictionary.
        save_path: Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(training_history['training_losses'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot evaluation rewards
    eval_iterations = np.arange(0, len(training_history['training_losses']), 100)
    ax2.plot(eval_iterations, training_history['eval_rewards'])
    ax2.set_title('Evaluation Rewards')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_learned_reward(
    irl_model: MaximumEntropyIRL,
    env: gym.Env,
    num_episodes: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """Evaluate the learned reward function.
    
    Args:
        irl_model: Trained IRL model.
        env: Gymnasium environment.
        num_episodes: Number of evaluation episodes.
        seed: Random seed.
        
    Returns:
        Evaluation metrics.
    """
    print(f"Evaluating learned reward function over {num_episodes} episodes...")
    
    # Set seed
    set_seed(seed)
    set_env_seed(env, seed)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        done = False
        while not done:
            # Get learned reward
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            learned_reward = irl_model.compute_reward(state_tensor).item()
            
            # Take random action (for evaluation)
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            
            episode_reward += learned_reward
            episode_length += 1
            
            done = terminated or truncated
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"  Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Maximum Entropy IRL')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--expert_trajectories', type=int, default=50, help='Number of expert trajectories')
    parser.add_argument('--policy_trajectories', type=int, default=50, help='Number of policy trajectories')
    parser.add_argument('--expert_type', type=str, default='heuristic', choices=['random', 'optimal', 'heuristic'], help='Expert policy type')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='assets', help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create environment
    print(f"Creating environment: {args.env}")
    env = gym.make(args.env)
    
    # Create configuration
    config = create_config()
    config.num_iterations = args.iterations
    
    # Collect expert data
    expert_trajectories = collect_expert_data(
        env,
        num_trajectories=args.expert_trajectories,
        expert_type=args.expert_type,
        seed=args.seed
    )
    
    # Collect policy data
    policy_trajectories = collect_policy_data(
        env,
        num_trajectories=args.policy_trajectories,
        seed=args.seed
    )
    
    # Train IRL model
    irl_model = train_irl_model(env, expert_trajectories, policy_trajectories, config)
    
    # Plot training results
    training_history = {
        'training_losses': irl_model.training_losses,
        'eval_rewards': irl_model.eval_rewards
    }
    plot_training_results(training_history, str(output_dir / 'training_results.png'))
    
    # Evaluate learned reward
    eval_metrics = evaluate_learned_reward(irl_model, env, seed=args.seed)
    
    # Save model
    model_path = output_dir / 'irl_model.pth'
    irl_model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Save evaluation metrics
    metrics_path = output_dir / 'eval_metrics.yaml'
    OmegaConf.save(eval_metrics, metrics_path)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    env.close()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
