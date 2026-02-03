"""Streamlit demo for Inverse Reinforcement Learning."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import gymnasium as gym
from pathlib import Path
import pandas as pd

from src.algorithms.max_entropy_irl import MaximumEntropyIRL, IRLConfig
from src.envs.data_collection import DataCollector, ExpertPolicy, RandomPolicy
from src.utils.seeding import set_seed


def load_model(model_path: str, state_dim: int) -> MaximumEntropyIRL:
    """Load a trained IRL model.
    
    Args:
        model_path: Path to the saved model.
        state_dim: State dimension.
        
    Returns:
        Loaded IRL model.
    """
    config = IRLConfig()
    model = MaximumEntropyIRL(state_dim, config)
    model.load_model(model_path)
    return model


def visualize_trajectory(
    env: gym.Env,
    policy,
    max_steps: int = 500,
    title: str = "Trajectory"
) -> tuple:
    """Visualize a trajectory and return states and rewards.
    
    Args:
        env: Gymnasium environment.
        policy: Policy function.
        max_steps: Maximum steps.
        title: Plot title.
        
    Returns:
        Tuple of (states, rewards, actions).
    """
    state, _ = env.reset()
    states = [state.copy()]
    rewards = []
    actions = []
    
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        states.append(next_state.copy())
        rewards.append(reward)
        actions.append(action)
        
        if terminated or truncated:
            break
        
        state = next_state
    
    return states, rewards, actions


def plot_reward_function(irl_model: MaximumEntropyIRL, env: gym.Env) -> go.Figure:
    """Plot the learned reward function.
    
    Args:
        irl_model: Trained IRL model.
        env: Gymnasium environment.
        
    Returns:
        Plotly figure.
    """
    # For CartPole, we'll plot reward vs pole angle and cart position
    pole_angles = np.linspace(-0.2, 0.2, 50)
    cart_positions = np.linspace(-2.4, 2.4, 50)
    
    rewards = np.zeros((len(pole_angles), len(cart_positions)))
    
    with torch.no_grad():
        for i, pole_angle in enumerate(pole_angles):
            for j, cart_pos in enumerate(cart_positions):
                # Create state: [cart_pos, cart_vel, pole_angle, pole_vel]
                state = np.array([cart_pos, 0.0, pole_angle, 0.0])
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                reward = irl_model.compute_reward(state_tensor).item()
                rewards[i, j] = reward
    
    fig = go.Figure(data=go.Heatmap(
        z=rewards,
        x=cart_positions,
        y=pole_angles,
        colorscale='RdBu',
        colorbar=dict(title="Learned Reward")
    ))
    
    fig.update_layout(
        title="Learned Reward Function",
        xaxis_title="Cart Position",
        yaxis_title="Pole Angle",
        width=600,
        height=500
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Inverse Reinforcement Learning Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Inverse Reinforcement Learning Demo")
    st.markdown("""
    This demo showcases Maximum Entropy Inverse Reinforcement Learning (IRL) 
    for learning reward functions from expert demonstrations.
    
    **⚠️ Safety Notice**: This is a research/educational demonstration. 
    Do not use for production control of real systems.
    """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        index=0
    )
    
    # Expert policy type
    expert_type = st.sidebar.selectbox(
        "Expert Policy Type",
        ["random", "heuristic", "optimal"],
        index=1
    )
    
    # Number of trajectories
    num_expert_trajectories = st.sidebar.slider(
        "Number of Expert Trajectories",
        min_value=10,
        max_value=100,
        value=50
    )
    
    num_policy_trajectories = st.sidebar.slider(
        "Number of Policy Trajectories",
        min_value=10,
        max_value=100,
        value=50
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    num_iterations = st.sidebar.slider(
        "Training Iterations",
        min_value=100,
        max_value=2000,
        value=1000
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=1e-4,
        max_value=1e-2,
        value=1e-3,
        format="%.0e"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training IRL Model")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training IRL model..."):
                # Set seed
                set_seed(seed)
                
                # Create environment
                env = gym.make(env_name)
                
                # Create configuration
                config = IRLConfig(
                    learning_rate=learning_rate,
                    num_iterations=num_iterations
                )
                
                # Collect expert data
                expert_policy = ExpertPolicy(env, policy_type=expert_type)
                collector = DataCollector(env, seed=seed)
                
                expert_trajectories = collector.collect_trajectories(
                    expert_policy,
                    num_trajectories=num_expert_trajectories,
                    max_steps=500
                )
                
                # Collect policy data
                random_policy = RandomPolicy(env)
                policy_trajectories = collector.collect_trajectories(
                    random_policy,
                    num_trajectories=num_policy_trajectories,
                    max_steps=500
                )
                
                # Train IRL model
                state_dim = env.observation_space.shape[0]
                irl_model = MaximumEntropyIRL(state_dim, config)
                
                expert_tuples = [traj.to_tuples() for traj in expert_trajectories]
                policy_tuples = [traj.to_tuples() for traj in policy_trajectories]
                
                training_history = irl_model.train(expert_tuples, policy_tuples)
                
                # Store in session state
                st.session_state.irl_model = irl_model
                st.session_state.training_history = training_history
                st.session_state.env = env
                st.session_state.expert_trajectories = expert_trajectories
                st.session_state.policy_trajectories = policy_trajectories
                
                st.success("Model trained successfully!")
                
                env.close()
    
    with col2:
        st.header("Model Status")
        
        if 'irl_model' in st.session_state:
            st.success("✅ Model Trained")
            
            # Display training metrics
            history = st.session_state.training_history
            final_loss = history['training_losses'][-1]
            final_reward = history['eval_rewards'][-1]
            
            st.metric("Final Loss", f"{final_loss:.4f}")
            st.metric("Final Reward", f"{final_reward:.4f}")
        else:
            st.info("No model trained yet")
    
    # Results section
    if 'irl_model' in st.session_state:
        st.header("Results")
        
        # Training curves
        st.subheader("Training Progress")
        
        history = st.session_state.training_history
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training Loss", "Evaluation Rewards"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot training loss
        fig.add_trace(
            go.Scatter(
                y=history['training_losses'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Plot evaluation rewards
        eval_iterations = np.arange(0, len(history['training_losses']), 100)
        fig.add_trace(
            go.Scatter(
                x=eval_iterations,
                y=history['eval_rewards'],
                mode='lines+markers',
                name='Eval Rewards',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Reward function visualization
        if env_name == "CartPole-v1":
            st.subheader("Learned Reward Function")
            reward_fig = plot_reward_function(st.session_state.irl_model, st.session_state.env)
            st.plotly_chart(reward_fig, use_container_width=True)
        
        # Trajectory comparison
        st.subheader("Trajectory Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Expert Trajectories**")
            expert_trajectories = st.session_state.expert_trajectories
            
            # Plot trajectory lengths
            lengths = [len(traj) for traj in expert_trajectories]
            fig_lengths = px.histogram(
                x=lengths,
                title="Expert Trajectory Lengths",
                labels={'x': 'Length', 'y': 'Count'}
            )
            st.plotly_chart(fig_lengths, use_container_width=True)
            
            st.write(f"Average length: {np.mean(lengths):.2f}")
            st.write(f"Total trajectories: {len(expert_trajectories)}")
        
        with col2:
            st.write("**Policy Trajectories**")
            policy_trajectories = st.session_state.policy_trajectories
            
            # Plot trajectory lengths
            lengths = [len(traj) for traj in policy_trajectories]
            fig_lengths = px.histogram(
                x=lengths,
                title="Policy Trajectory Lengths",
                labels={'x': 'Length', 'y': 'Count'}
            )
            st.plotly_chart(fig_lengths, use_container_width=True)
            
            st.write(f"Average length: {np.mean(lengths):.2f}")
            st.write(f"Total trajectories: {len(policy_trajectories)}")
        
        # Live evaluation
        st.subheader("Live Evaluation")
        
        if st.button("🎯 Run Evaluation"):
            with st.spinner("Running evaluation..."):
                env = st.session_state.env
                irl_model = st.session_state.irl_model
                
                # Run evaluation episodes
                episode_rewards = []
                episode_lengths = []
                
                for episode in range(10):
                    state, _ = env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    
                    done = False
                    while not done:
                        # Get learned reward
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        learned_reward = irl_model.compute_reward(state_tensor).item()
                        
                        # Take random action
                        action = env.action_space.sample()
                        next_state, _, terminated, truncated, _ = env.step(action)
                        
                        episode_reward += learned_reward
                        episode_length += 1
                        
                        done = terminated or truncated
                        state = next_state
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Mean Reward",
                        f"{np.mean(episode_rewards):.4f}",
                        f"±{np.std(episode_rewards):.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Mean Length",
                        f"{np.mean(episode_lengths):.2f}",
                        f"±{np.std(episode_lengths):.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Success Rate",
                        f"{np.mean([l > 100 for l in episode_lengths]) * 100:.1f}%"
                    )
                
                # Plot episode results
                fig_episodes = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Episode Rewards", "Episode Lengths")
                )
                
                fig_episodes.add_trace(
                    go.Scatter(
                        y=episode_rewards,
                        mode='lines+markers',
                        name='Rewards',
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
                
                fig_episodes.add_trace(
                    go.Scatter(
                        y=episode_lengths,
                        mode='lines+markers',
                        name='Lengths',
                        line=dict(color='orange')
                    ),
                    row=1, col=2
                )
                
                fig_episodes.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_episodes, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About Maximum Entropy IRL**: This algorithm learns a reward function that maximizes 
    the entropy of the policy while matching expert demonstrations. It's particularly 
    useful when the true reward function is unknown but expert behavior is available.
    """)


if __name__ == "__main__":
    main()
