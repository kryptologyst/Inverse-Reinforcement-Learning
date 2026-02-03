"""Tests for Maximum Entropy IRL implementation."""

import pytest
import numpy as np
import torch
import gymnasium as gym
from unittest.mock import Mock

from src.algorithms.max_entropy_irl import MaximumEntropyIRL, IRLConfig, RewardNetwork
from src.envs.data_collection import DataCollector, ExpertPolicy, RandomPolicy, Trajectory
from src.utils.seeding import set_seed, get_device


class TestRewardNetwork:
    """Test RewardNetwork class."""
    
    def test_initialization(self):
        """Test network initialization."""
        config = IRLConfig(hidden_size=32, num_layers=1)
        network = RewardNetwork(state_dim=4, config=config)
        
        assert isinstance(network, RewardNetwork)
        assert network.device == get_device()
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = IRLConfig(hidden_size=32, num_layers=1)
        network = RewardNetwork(state_dim=4, config=config)
        
        # Test single state
        state = torch.randn(1, 4)
        reward = network(state)
        
        assert reward.shape == (1, 1)
        assert isinstance(reward, torch.Tensor)
        
        # Test batch of states
        states = torch.randn(5, 4)
        rewards = network(states)
        
        assert rewards.shape == (5, 1)


class TestMaximumEntropyIRL:
    """Test MaximumEntropyIRL class."""
    
    def test_initialization(self):
        """Test IRL model initialization."""
        config = IRLConfig()
        model = MaximumEntropyIRL(state_dim=4, config=config)
        
        assert isinstance(model, MaximumEntropyIRL)
        assert isinstance(model.reward_network, RewardNetwork)
        assert len(model.training_losses) == 0
        assert len(model.eval_rewards) == 0
    
    def test_compute_reward(self):
        """Test reward computation."""
        config = IRLConfig()
        model = MaximumEntropyIRL(state_dim=4, config=config)
        
        states = torch.randn(3, 4)
        rewards = model.compute_reward(states)
        
        assert rewards.shape == (3, 1)
        assert isinstance(rewards, torch.Tensor)
    
    def test_train_step(self):
        """Test single training step."""
        config = IRLConfig(num_iterations=1)
        model = MaximumEntropyIRL(state_dim=4, config=config)
        
        # Create dummy trajectories
        expert_trajectories = [
            [(np.random.randn(4), 0), (np.random.randn(4), 1)],
            [(np.random.randn(4), 1), (np.random.randn(4), 0)]
        ]
        
        policy_trajectories = [
            [(np.random.randn(4), 0), (np.random.randn(4), 1)],
            [(np.random.randn(4), 1), (np.random.randn(4), 0)]
        ]
        
        loss = model.train_step(expert_trajectories, policy_trajectories)
        
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative


class TestDataCollection:
    """Test data collection utilities."""
    
    def test_trajectory_creation(self):
        """Test Trajectory class."""
        states = [np.random.randn(4) for _ in range(5)]
        actions = [0, 1, 0, 1, 0]
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0]
        dones = [False, False, False, False, True]
        infos = [{} for _ in range(5)]
        
        traj = Trajectory(states, actions, rewards, dones, infos)
        
        assert len(traj) == 5
        assert len(traj.to_tuples()) == 5
        assert traj.to_tuples()[0] == (states[0], actions[0])
    
    def test_expert_policy(self):
        """Test ExpertPolicy class."""
        env = gym.make("CartPole-v1")
        
        # Test random policy
        random_policy = ExpertPolicy(env, policy_type="random")
        state = np.random.randn(4)
        action = random_policy(state)
        
        assert action in [0, 1]
        
        # Test heuristic policy
        heuristic_policy = ExpertPolicy(env, policy_type="heuristic")
        action = heuristic_policy(state)
        
        assert action in [0, 1]
        
        env.close()
    
    def test_random_policy(self):
        """Test RandomPolicy class."""
        env = gym.make("CartPole-v1")
        policy = RandomPolicy(env)
        
        state = np.random.randn(4)
        action = policy(state)
        
        assert action in [0, 1]
        
        env.close()
    
    def test_data_collector(self):
        """Test DataCollector class."""
        env = gym.make("CartPole-v1")
        collector = DataCollector(env, seed=42)
        
        # Create a simple policy
        def simple_policy(state):
            return 0 if state[0] < 0 else 1
        
        # Collect a single trajectory
        traj = collector.collect_trajectory(simple_policy, max_steps=10)
        
        assert isinstance(traj, Trajectory)
        assert len(traj) > 0
        assert len(traj.states) == len(traj.actions) + 1  # States include initial state
        
        env.close()


class TestSeeding:
    """Test seeding utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.rand()
        
        np.random.seed(42)
        val2 = np.random.rand()
        
        assert val1 == val2
        
        # Test torch
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


@pytest.fixture
def cartpole_env():
    """Create CartPole environment for testing."""
    env = gym.make("CartPole-v1")
    yield env
    env.close()


@pytest.fixture
def irl_config():
    """Create IRL configuration for testing."""
    return IRLConfig(
        hidden_size=32,
        num_layers=1,
        learning_rate=0.01,
        num_iterations=10,
        batch_size=4
    )


def test_integration(cartpole_env, irl_config):
    """Test end-to-end integration."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Create policies
    expert_policy = ExpertPolicy(cartpole_env, policy_type="heuristic")
    random_policy = RandomPolicy(cartpole_env)
    
    # Create data collector
    collector = DataCollector(cartpole_env, seed=42)
    
    # Collect trajectories
    expert_trajectories = collector.collect_trajectories(
        expert_policy, num_trajectories=5, max_steps=50
    )
    
    policy_trajectories = collector.collect_trajectories(
        random_policy, num_trajectories=5, max_steps=50
    )
    
    # Create and train IRL model
    state_dim = cartpole_env.observation_space.shape[0]
    model = MaximumEntropyIRL(state_dim, irl_config)
    
    # Convert to tuples
    expert_tuples = [traj.to_tuples() for traj in expert_trajectories]
    policy_tuples = [traj.to_tuples() for traj in policy_trajectories]
    
    # Train
    history = model.train(expert_tuples, policy_tuples)
    
    # Check results
    assert len(history["training_losses"]) == irl_config.num_iterations
    assert len(history["eval_rewards"]) > 0
    assert isinstance(history["training_losses"][0], float)
    assert isinstance(history["eval_rewards"][0], float)


if __name__ == "__main__":
    pytest.main([__file__])
