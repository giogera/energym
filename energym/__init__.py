"""Energym: Gymnasium environment for prosumer energy nodes with battery storage."""

import gymnasium as gym

from energym.envs import EnergyStorageEnv

__version__ = '0.1.2'
__all__ = ['EnergyStorageEnv']


# register the environment
def _register_env() -> None:
    """Register the EnergyStorage environment with Gymnasium."""
    gym.register(
        id='energym_env/EnergyStorage-v0',
        entry_point='energym.envs:EnergyStorageEnv'
    )


_register_env()
