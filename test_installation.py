#!/usr/bin/env python3
"""Test script to verify IRL installation and basic functionality."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import gymnasium as gym

from algorithms.max_entropy_irl import MaximumEntropyIRL, IRLConfig
from envs.data_collection import DataCollector, ExpertPolicy, RandomPolicy
from utils.seeding import set_seed


def test_basic_functionality():
    """Test basic IRL functionality."""
    print("Testing basic IRL functionality...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = gym.make("CartPole-v1")
    print(f"✓ Created environment: {env.spec.id}")
    
    # Create policies
    expert_policy = ExpertPolicy(env, policy_type="heuristic")
    random_policy = RandomPolicy(env)
    print("✓ Created expert and random policies")
    
    # Create data collector
    collector = DataCollector(env, seed=42)
    print("✓ Created data collector")
    
    # Collect small number of trajectories
    print("Collecting trajectories...")
    expert_trajectories = collector.collect_trajectories(
        expert_policy, num_trajectories=5, max_steps=50
    )
    policy_trajectories = collector.collect_trajectories(
        random_policy, num_trajectories=5, max_steps=50
    )
    print(f"✓ Collected {len(expert_trajectories)} expert and {len(policy_trajectories)} policy trajectories")
    
    # Create IRL model
    state_dim = env.observation_space.shape[0]
    config = IRLConfig(
        hidden_size=32,
        num_layers=1,
        learning_rate=0.01,
        num_iterations=10,  # Very short training for test
        batch_size=2
    )
    irl_model = MaximumEntropyIRL(state_dim, config)
    print("✓ Created IRL model")
    
    # Convert trajectories to tuples
    expert_tuples = [traj.to_tuples() for traj in expert_trajectories]
    policy_tuples = [traj.to_tuples() for traj in policy_trajectories]
    
    # Train model
    print("Training IRL model...")
    training_history = irl_model.train(expert_tuples, policy_tuples)
    print("✓ Training completed")
    
    # Test reward computation
    test_state = torch.randn(1, state_dim)
    reward = irl_model.compute_reward(test_state)
    print(f"✓ Reward computation works: {reward.item():.4f}")
    
    # Test evaluation
    eval_reward = irl_model.evaluate(expert_tuples[:2])
    print(f"✓ Evaluation works: {eval_reward:.4f}")
    
    env.close()
    print("\n🎉 All tests passed! IRL installation is working correctly.")


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    from utils.seeding import get_device
    device = get_device()
    print(f"✓ Detected device: {device}")
    
    # Test PyTorch operations on device
    x = torch.randn(2, 3).to(device)
    y = torch.randn(3, 4).to(device)
    z = torch.mm(x, y)
    print(f"✓ PyTorch operations work on {device}")


def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
        
        import torch
        print("✓ torch")
        
        import gymnasium as gym
        print("✓ gymnasium")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
        
        import pandas as pd
        print("✓ pandas")
        
        import streamlit as st
        print("✓ streamlit")
        
        import plotly.graph_objects as go
        print("✓ plotly")
        
        from omegaconf import OmegaConf
        print("✓ omegaconf")
        
        print("✓ All required packages imported successfully")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("INVERSE REINFORCEMENT LEARNING - INSTALLATION TEST")
    print("=" * 60)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        sys.exit(1)
    
    # Test device detection
    test_device_detection()
    
    # Test basic functionality
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - READY TO USE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run training: python scripts/quick_start.py")
    print("2. Launch demo: python scripts/run_demo.py")
    print("3. Run tests: python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
