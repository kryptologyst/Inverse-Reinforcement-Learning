"""Maximum Entropy Inverse Reinforcement Learning implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from ..utils.seeding import get_device


@dataclass
class IRLConfig:
    """Configuration for Maximum Entropy IRL."""
    
    # Model architecture
    hidden_size: int = 64
    num_layers: int = 2
    
    # Training parameters
    learning_rate: float = 1e-3
    num_iterations: int = 1000
    batch_size: int = 32
    
    # IRL specific parameters
    regularization_weight: float = 0.01
    entropy_weight: float = 1.0
    
    # Evaluation
    eval_frequency: int = 100
    num_eval_episodes: int = 10


class RewardNetwork(nn.Module):
    """Neural network for reward function approximation."""
    
    def __init__(self, state_dim: int, config: IRLConfig) -> None:
        """Initialize reward network.
        
        Args:
            state_dim: Dimension of state space.
            config: IRL configuration.
        """
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = config.hidden_size
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through reward network.
        
        Args:
            states: Batch of states.
            
        Returns:
            Predicted rewards.
        """
        return self.network(states)


class MaximumEntropyIRL:
    """Maximum Entropy Inverse Reinforcement Learning algorithm."""
    
    def __init__(self, state_dim: int, config: IRLConfig) -> None:
        """Initialize Maximum Entropy IRL.
        
        Args:
            state_dim: Dimension of state space.
            config: IRL configuration.
        """
        self.config = config
        self.device = get_device()
        
        # Initialize reward network
        self.reward_network = RewardNetwork(state_dim, config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.reward_network.parameters(),
            lr=config.learning_rate
        )
        
        # Training history
        self.training_losses: List[float] = []
        self.eval_rewards: List[float] = []
    
    def compute_reward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute rewards for given states.
        
        Args:
            states: Batch of states.
            
        Returns:
            Predicted rewards.
        """
        return self.reward_network(states)
    
    def compute_policy_loss(
        self,
        expert_trajectories: List[List[Tuple[np.ndarray, int]]],
        policy_trajectories: List[List[Tuple[np.ndarray, int]]]
    ) -> torch.Tensor:
        """Compute policy loss for Maximum Entropy IRL.
        
        Args:
            expert_trajectories: Expert demonstration trajectories.
            policy_trajectories: Policy-generated trajectories.
            
        Returns:
            Policy loss.
        """
        # Compute expert rewards
        expert_rewards = []
        for traj in expert_trajectories:
            states = torch.tensor([s for s, _ in traj], dtype=torch.float32).to(self.device)
            rewards = self.compute_reward(states)
            expert_rewards.append(rewards.sum())
        
        # Compute policy rewards
        policy_rewards = []
        for traj in policy_trajectories:
            states = torch.tensor([s for s, _ in traj], dtype=torch.float32).to(self.device)
            rewards = self.compute_reward(states)
            policy_rewards.append(rewards.sum())
        
        # Convert to tensors
        expert_rewards = torch.stack(expert_rewards)
        policy_rewards = torch.stack(policy_rewards)
        
        # Maximum entropy loss: maximize expert reward, minimize policy reward
        loss = policy_rewards.mean() - expert_rewards.mean()
        
        # Add regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.reward_network.parameters())
        loss += self.config.regularization_weight * l2_reg
        
        return loss
    
    def train_step(
        self,
        expert_trajectories: List[List[Tuple[np.ndarray, int]]],
        policy_trajectories: List[List[Tuple[np.ndarray, int]]]
    ) -> float:
        """Perform one training step.
        
        Args:
            expert_trajectories: Expert demonstration trajectories.
            policy_trajectories: Policy-generated trajectories.
            
        Returns:
            Training loss.
        """
        self.optimizer.zero_grad()
        
        loss = self.compute_policy_loss(expert_trajectories, policy_trajectories)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        expert_trajectories: List[List[Tuple[np.ndarray, int]]],
        policy_trajectories: List[List[Tuple[np.ndarray, int]]]
    ) -> Dict[str, List[float]]:
        """Train the IRL model.
        
        Args:
            expert_trajectories: Expert demonstration trajectories.
            policy_trajectories: Policy-generated trajectories.
            
        Returns:
            Training history.
        """
        self.reward_network.train()
        
        for iteration in range(self.config.num_iterations):
            # Sample batches
            expert_batch = np.random.choice(
                expert_trajectories,
                size=min(self.config.batch_size, len(expert_trajectories)),
                replace=False
            ).tolist()
            
            policy_batch = np.random.choice(
                policy_trajectories,
                size=min(self.config.batch_size, len(policy_trajectories)),
                replace=False
            ).tolist()
            
            # Training step
            loss = self.train_step(expert_batch, policy_batch)
            self.training_losses.append(loss)
            
            # Evaluation
            if iteration % self.config.eval_frequency == 0:
                eval_reward = self.evaluate(expert_trajectories[:self.config.num_eval_episodes])
                self.eval_rewards.append(eval_reward)
                
                print(f"Iteration {iteration}: Loss = {loss:.4f}, Eval Reward = {eval_reward:.4f}")
        
        return {
            "training_losses": self.training_losses,
            "eval_rewards": self.eval_rewards
        }
    
    def evaluate(self, trajectories: List[List[Tuple[np.ndarray, int]]]) -> float:
        """Evaluate the learned reward function.
        
        Args:
            trajectories: Trajectories to evaluate.
            
        Returns:
            Average reward.
        """
        self.reward_network.eval()
        
        total_reward = 0.0
        with torch.no_grad():
            for traj in trajectories:
                states = torch.tensor([s for s, _ in traj], dtype=torch.float32).to(self.device)
                rewards = self.compute_reward(states)
                total_reward += rewards.sum().item()
        
        return total_reward / len(trajectories)
    
    def save_model(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model.
        """
        torch.save({
            "reward_network": self.reward_network.state_dict(),
            "config": self.config,
            "training_losses": self.training_losses,
            "eval_rewards": self.eval_rewards
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model.
        
        Args:
            path: Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.reward_network.load_state_dict(checkpoint["reward_network"])
        self.training_losses = checkpoint["training_losses"]
        self.eval_rewards = checkpoint["eval_rewards"]
