"""Utility functions for reproducible experiments and device management."""

import random
from typing import Optional, Union

import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_env_seed(env: gym.Env, seed: int) -> None:
    """Set seed for gymnasium environment.
    
    Args:
        env: Gymnasium environment.
        seed: Random seed value.
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
