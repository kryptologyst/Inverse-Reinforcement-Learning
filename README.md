# Inverse Reinforcement Learning (IRL) Project

A research-ready implementation of Maximum Entropy Inverse Reinforcement Learning for learning reward functions from expert demonstrations.

## Overview

This project implements Maximum Entropy IRL, a state-of-the-art algorithm for inferring reward functions from observed expert behavior. Instead of learning policies directly from environment rewards, IRL learns what reward function would make the observed expert behavior optimal.

### Key Features

- **Maximum Entropy IRL**: Implements the principled maximum entropy approach to reward learning
- **Modern Stack**: Built with PyTorch 2.x, Gymnasium, and modern Python practices
- **Reproducible**: Deterministic seeding across all random number generators
- **Interactive Demo**: Streamlit-based visualization and experimentation
- **Comprehensive Testing**: Full test suite with integration tests
- **Production Ready**: Clean code structure with type hints and documentation

## Safety Notice

**⚠️ IMPORTANT**: This is a research and educational demonstration. Do not use for production control of real-world systems, especially in safety-critical domains like robotics, healthcare, finance, or energy systems.

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- MPS support (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Inverse-Reinforcement-Learning.git
cd Inverse-Reinforcement-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. Verify installation:
```bash
python -m pytest tests/ -v
```

## Quick Start

### Command Line Training

Train an IRL model with default settings:

```bash
python scripts/quick_start.py
```

Customize training parameters:

```bash
python scripts/quick_start.py \
    --env CartPole-v1 \
    --expert_trajectories 100 \
    --policy_trajectories 100 \
    --expert_type heuristic \
    --iterations 2000 \
    --seed 42
```

### Interactive Demo

Launch the Streamlit demo for interactive experimentation:

```bash
python scripts/run_demo.py
```

The demo will open in your browser at `http://localhost:8501`.

### Programmatic Usage

```python
import gymnasium as gym
from src.algorithms.max_entropy_irl import MaximumEntropyIRL, IRLConfig
from src.envs.data_collection import DataCollector, ExpertPolicy, RandomPolicy
from src.utils.seeding import set_seed

# Set seed for reproducibility
set_seed(42)

# Create environment
env = gym.make("CartPole-v1")

# Create policies
expert_policy = ExpertPolicy(env, policy_type="heuristic")
random_policy = RandomPolicy(env)

# Collect demonstrations
collector = DataCollector(env, seed=42)
expert_trajectories = collector.collect_trajectories(expert_policy, num_trajectories=50)
policy_trajectories = collector.collect_trajectories(random_policy, num_trajectories=50)

# Train IRL model
config = IRLConfig(num_iterations=1000)
state_dim = env.observation_space.shape[0]
irl_model = MaximumEntropyIRL(state_dim, config)

expert_tuples = [traj.to_tuples() for traj in expert_trajectories]
policy_tuples = [traj.to_tuples() for traj in policy_trajectories]

training_history = irl_model.train(expert_tuples, policy_tuples)

# Evaluate learned reward
eval_reward = irl_model.evaluate(expert_tuples[:10])
print(f"Evaluation reward: {eval_reward:.4f}")

env.close()
```

## Project Structure

```
├── src/                          # Source code
│   ├── algorithms/               # IRL algorithms
│   │   ├── __init__.py
│   │   └── max_entropy_irl.py    # Maximum Entropy IRL implementation
│   ├── envs/                     # Environment utilities
│   │   ├── __init__.py
│   │   └── data_collection.py    # Data collection and policies
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── seeding.py            # Reproducibility utilities
│   └── train/                    # Training scripts
│       └── train_irl.py          # Main training script
├── configs/                      # Configuration files
│   ├── default.yaml              # Default configuration
│   └── test.yaml                 # Test configuration
├── scripts/                      # Executable scripts
│   ├── quick_start.py            # Quick start training
│   └── run_demo.py               # Launch demo
├── demo/                         # Interactive demo
│   └── app.py                    # Streamlit demo
├── tests/                        # Test suite
│   └── test_irl.py               # Comprehensive tests
├── assets/                       # Output directory (created during training)
├── data/                         # Data directory
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files for experiment management. Key parameters:

### Environment Settings
- `env.name`: Gymnasium environment name
- `env.max_steps`: Maximum steps per episode
- `env.seed`: Environment random seed

### Data Collection
- `data.num_expert_trajectories`: Number of expert demonstrations
- `data.num_policy_trajectories`: Number of policy trajectories for comparison
- `data.expert_type`: Expert policy type (random, optimal, heuristic)

### Model Architecture
- `model.hidden_size`: Hidden layer size
- `model.num_layers`: Number of hidden layers
- `model.dropout`: Dropout rate

### Training Parameters
- `training.learning_rate`: Learning rate
- `training.num_iterations`: Number of training iterations
- `training.batch_size`: Batch size
- `training.regularization_weight`: L2 regularization weight

## Environments

The project supports various Gymnasium environments:

- **CartPole-v1**: Classic control task (default)
- **MountainCar-v0**: Sparse reward navigation
- **Acrobot-v1**: Underactuated swing-up task

### Expert Policies

Three types of expert policies are implemented:

1. **Random**: Random action selection (baseline)
2. **Heuristic**: Rule-based policy using domain knowledge
3. **Optimal**: Simplified optimal policy (CartPole-specific)

## Evaluation Metrics

The project provides comprehensive evaluation:

### Learning Metrics
- Training loss over iterations
- Evaluation rewards during training
- Convergence analysis

### Performance Metrics
- Mean episode reward ± standard deviation
- Mean episode length ± standard deviation
- Success rate (episodes exceeding threshold)

### Reward Function Analysis
- Learned reward function visualization
- Reward landscape plots
- Comparison with true rewards

## Reproducibility

The project ensures full reproducibility:

- **Deterministic Seeding**: All random number generators are seeded
- **Device Agnostic**: Automatic CUDA/MPS/CPU fallback
- **Version Pinning**: Exact dependency versions specified
- **Configuration Management**: All hyperparameters in config files

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_irl.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Development

### Code Quality

The project enforces high code quality:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis
- **Testing**: Comprehensive test coverage

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `black src/ tests/` and `ruff check src/ tests/`
5. Submit a pull request

## Advanced Usage

### Custom Environments

To use custom environments:

```python
import gymnasium as gym
from gymnasium import register

# Register custom environment
register(
    id='CustomEnv-v0',
    entry_point='path.to.custom.env:CustomEnv'
)

# Use in training
env = gym.make('CustomEnv-v0')
```

### Custom Expert Policies

Implement custom expert policies:

```python
from src.envs.data_collection import ExpertPolicy

class CustomExpertPolicy(ExpertPolicy):
    def __init__(self, env):
        super().__init__(env, policy_type="custom")
    
    def _custom_action(self, state):
        # Implement custom policy logic
        return action
```

### Hyperparameter Tuning

Use configuration files for hyperparameter sweeps:

```bash
# Train with different configurations
python scripts/quick_start.py --config configs/experiment1.yaml
python scripts/quick_start.py --config configs/experiment2.yaml
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Environment Not Found**: Install required environment packages
3. **Import Errors**: Ensure src/ is in Python path
4. **Demo Won't Launch**: Check Streamlit installation

### Performance Tips

1. **GPU Acceleration**: Use CUDA for faster training
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Vectorized Environments**: Use parallel data collection
4. **Model Checkpointing**: Save models during training

## Citation

If you use this code in your research, please cite:

```bibtex
@software{inverse_reinforcement_learning,
  title={Maximum Entropy Inverse Reinforcement Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Inverse-Reinforcement-Learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Maximum Entropy IRL paper by Ziebart et al.
- Gymnasium team for the excellent RL environment interface
- PyTorch team for the deep learning framework
- Streamlit team for the interactive demo framework

## References

1. Ziebart, B., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement learning. AAAI.
2. Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. NeurIPS.
3. Finn, C., Levine, S., & Abbeel, P. (2016). Guided cost learning: Deep inverse optimal control via policy optimization. ICML.
# Inverse-Reinforcement-Learning
