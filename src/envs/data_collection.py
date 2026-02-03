"""Environment utilities and data collection for IRL."""

import gymnasium as gym
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import torch

from ..utils.seeding import set_env_seed


@dataclass
class Trajectory:
    """Represents a single trajectory."""
    
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    
    def __len__(self) -> int:
        """Return trajectory length."""
        return len(self.states)
    
    def to_tuples(self) -> List[Tuple[np.ndarray, int]]:
        """Convert to list of (state, action) tuples."""
        return list(zip(self.states, self.actions))


class DataCollector:
    """Collects trajectories from environments."""
    
    def __init__(self, env: gym.Env, seed: Optional[int] = None) -> None:
        """Initialize data collector.
        
        Args:
            env: Gymnasium environment.
            seed: Random seed for environment.
        """
        self.env = env
        self.seed = seed
        
        if seed is not None:
            set_env_seed(env, seed)
    
    def collect_trajectory(
        self,
        policy: Callable[[np.ndarray], int],
        max_steps: int = 1000
    ) -> Trajectory:
        """Collect a single trajectory using the given policy.
        
        Args:
            policy: Policy function that takes state and returns action.
            max_steps: Maximum number of steps per trajectory.
            
        Returns:
            Collected trajectory.
        """
        state, _ = self.env.reset()
        
        states = [state.copy()]
        actions = []
        rewards = []
        dones = []
        infos = []
        
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
            
            if terminated or truncated:
                break
            
            state = next_state
            states.append(state.copy())
        
        return Trajectory(states, actions, rewards, dones, infos)
    
    def collect_trajectories(
        self,
        policy: Callable[[np.ndarray], int],
        num_trajectories: int,
        max_steps: int = 1000
    ) -> List[Trajectory]:
        """Collect multiple trajectories.
        
        Args:
            policy: Policy function.
            num_trajectories: Number of trajectories to collect.
            max_steps: Maximum steps per trajectory.
            
        Returns:
            List of collected trajectories.
        """
        trajectories = []
        
        for _ in range(num_trajectories):
            traj = self.collect_trajectory(policy, max_steps)
            trajectories.append(traj)
        
        return trajectories


class ExpertPolicy:
    """Expert policy for generating demonstrations."""
    
    def __init__(self, env: gym.Env, policy_type: str = "random") -> None:
        """Initialize expert policy.
        
        Args:
            env: Gymnasium environment.
            policy_type: Type of policy ("random", "optimal", "heuristic").
        """
        self.env = env
        self.policy_type = policy_type
        
        if policy_type == "optimal":
            self._setup_optimal_policy()
        elif policy_type == "heuristic":
            self._setup_heuristic_policy()
    
    def _setup_optimal_policy(self) -> None:
        """Setup optimal policy (placeholder for now)."""
        # For CartPole, optimal policy would balance the pole
        # This is a simplified version
        pass
    
    def _setup_heuristic_policy(self) -> None:
        """Setup heuristic policy."""
        # For CartPole, heuristic could be: move in direction of pole angle
        pass
    
    def __call__(self, state: np.ndarray) -> int:
        """Get action from policy.
        
        Args:
            state: Current state.
            
        Returns:
            Action to take.
        """
        if self.policy_type == "random":
            return self.env.action_space.sample()
        elif self.policy_type == "optimal":
            return self._optimal_action(state)
        elif self.policy_type == "heuristic":
            return self._heuristic_action(state)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def _optimal_action(self, state: np.ndarray) -> int:
        """Get optimal action (simplified for CartPole).
        
        Args:
            state: Current state.
            
        Returns:
            Optimal action.
        """
        # Simplified optimal policy for CartPole
        # Move cart in direction opposite to pole angle
        pole_angle = state[2]  # Third element is pole angle
        return 0 if pole_angle < 0 else 1
    
    def _heuristic_action(self, state: np.ndarray) -> int:
        """Get heuristic action.
        
        Args:
            state: Current state.
            
        Returns:
            Heuristic action.
        """
        # Heuristic: move cart towards center and balance pole
        cart_position = state[0]
        pole_angle = state[2]
        
        # Move towards center
        if abs(cart_position) > 0.1:
            return 0 if cart_position < 0 else 1
        
        # Balance pole
        return 0 if pole_angle < 0 else 1


class RandomPolicy:
    """Random policy for comparison."""
    
    def __init__(self, env: gym.Env) -> None:
        """Initialize random policy.
        
        Args:
            env: Gymnasium environment.
        """
        self.env = env
    
    def __call__(self, state: np.ndarray) -> int:
        """Get random action.
        
        Args:
            state: Current state (ignored).
            
        Returns:
            Random action.
        """
        return self.env.action_space.sample()
