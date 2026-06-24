# EnergyGym

Physics-based Gymnasium environment for simulating prosumer energy nodes with battery energy storage systems (BESS).

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## Features

- Gymnasium-compatible environment registered as `energym_env/EnergyStorage-v0`.
- Physics-based battery model with commercial datasheet parameters. 
- Market integration using time series of energy buy/sell prices.
- Helper modules for observation building, battery dynamics, and reward calculation.
- Built-in MILP solver for benchmarking.

## Quick Start

### Installation

```bash
# create Conda environment
conda create -n energym python=3.12 -y

# clone repo and install
git clone https://github.com/giogera/energym.git
cd energym
pip install -e .
```

### Basic Usage

```python
import gymnasium as gym
import energym

# create environment
env = gym.make(
    'energym_env/EnergyStorage-v0',
    csv_path='dataset/prosumer.csv',
)

# reset and run episode
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random action
    obs, reward, done, truncated, info = env.step(action)

print(f"Episode reward: {info['episode_reward']:.2f}")
```

## Environment Specification

### Observation Space

12-dimensional vector containing:
- **time features** (6): cyclical encoding of month, day (within month), hour
- **state** (1): battery state of charge (SOC)
- **market data** (4): normalized production, consumption, buy price, sell price
- **holiday flag** (1): weekend/Italian holiday indicator

### Action Space

Continuous 1-dimensional action in `[-1.0, 1.0]`:
- **positive**: charge battery (scaled to battery nominal power)
- **negative**: discharge battery (scaled to battery nominal power)
- **zero**: idle

### Reward

Calculated from energy arbitrage and constraint violations:
```
reward = net_profit - violation_penalty

where:
  net_profit = sell_price * export - buy_price * import 
             - usage_price * soc_variation
  violation_penalty = violation_weight * (constraint_violation)^2
```

### Data Format

CSV file with columns:
```
datetime,production,consumption,buy_price,sell_price
2023-01-01 00:00,0.0,0.5,0.25,0.15
2023-01-01 01:00,0.0,0.4,0.25,0.15
...
```

## Project Structure

```
energym/
├── energym/
│   ├── __init__.py
│   ├── config.py               # configuration and defaults
│   ├── utils.py                # utility functions
│   ├── solvers/
│   │   ├── __init__.py
│   │   └── milp.py             # MILP solver for optimal actions
│   └── envs/
│       ├── __init__.py
│       ├── energym_env.py      # main EnergyStorageEnv
│       ├── observations.py     # observation builder
│       ├── physics.py          # battery physics and constraints
│       └── rewards.py          # reward calculation
├── dataset/
│   └── ...                     # time series data
├── pyproject.toml              # package configuration
├── README.md                   # documentation
├── CHANGELOG.md                # changelog
└── .gitignore
```

## Configuration

Edit `energym/config.py` to customize:

```python
# battery parameters
DEFAULT_BATTERY_CONFIG = {
    'capacity': 20.0,
    'efficiency_charge': 0.95,
    'efficiency_discharge': 0.95,
    'power_charge_max': 2.5,
    'power_discharge_max': 2.5,
    'dod_max': 90.0,  # depth of discharge %
}

# reward parameters
DEFAULT_REWARD_CONFIG = {
    'usage_price': 0.001,  # €/kWh
    'violation_penalty': 0.01,
}

# other settings
DEFAULT_DELTA_T = 1.0  # hour
INITIAL_SOC = 0.5      # 50%
```

## Examples

### Using a Rule-Based Controller

```python
import gymnasium as gym
import energym

# create environment
env = gym.make(
    'energym_env/EnergyStorage-v0',
    csv_path='dataset/prosumer.csv',
)
p_nom = env.unwrapped.p_nom

# rule-based controller
def rbc(production, consumption, p_nom):
    """Rule-based controller for energy storage.

    Args:
        production: Production at current step in kW.
        consumption: Consumption at current step in kW.
        p_nom: Nominal power of the battery in kW.

    Returns:
        Normalized action in [-1, 1].
    """
    balance = production - consumption
    return balance / p_nom

# evaluate
obs, info = env.reset()
episode_reward = 0
total_profit = 0
done = False
while not done:
    action = rbc(info['production'], info['consumption'], p_nom)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    total_profit += info['profit']

print(f"Episode reward: {episode_reward:.2f}")
print(f"Total profit: {total_profit:.2f}")
```

### Using Reinforcement Learning (SAC)

```python
import gymnasium as gym
from stable_baselines3 import SAC
import energym

# create environment
env = gym.make(
    'energym_env/EnergyStorage-v0',
    csv_path='dataset/prosumer.csv',
)

# train RL agent
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# evaluate
obs, info = env.reset()
episode_reward = 0
total_profit = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    total_profit += info['profit']

print(f"Episode reward: {episode_reward:.2f}")
print(f"Total profit: {total_profit:.2f}")
```

### Using MILP solver

```python
import gymnasium as gym
import energym

# create environment
env = gym.make(
    'energym_env/EnergyStorage-v0',
    csv_path='dataset/prosumer.csv',
)

# run MILP solver
optimal_actions, solution_df = env.unwrapped.milp_solution()

# evaluate
obs, info = env.reset()
episode_reward = 0
total_profit = 0
done = False
while not done:
    action = optimal_actions[info['timestep']]
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    total_profit += info['profit']

print(f"Episode reward: {episode_reward:.2f}")
print(f"Total profit: {total_profit:.2f}")
```

## Citation

If you use this environment in your research, please cite:
```bibtex
@software{energym2026,
  title={EnergyGym: Gymnasium environment for energy management in prosumer nodes},
  author={Giovanni Geraci},
  email={giovanni.geraci2@unisi.it},
  year={2026},
  url={https://github.com/giogera/energym}
}
```

## License

MIT License - see [LICENCE](LICENCE) for details