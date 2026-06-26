"""Gymnasium environment for prosumer energy node with battery storage."""

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from energym import config, utils
from energym.envs.observations import ObservationBuilder
from energym.envs.physics import BatteryPhysics
from energym.envs.rewards import RewardCalculator
from energym.solvers import milp_solver


class EnergyStorageEnv(gym.Env):
    """Gymnasium environment simulating a prosumer energy node with battery storage.

    The environment simulates a prosumer with solar production, consumption, and a battery
    that can be charged/discharged to maximize energy arbitrage profit. The agent controls
    the net battery power (normalized to [-1, 1]), which is scaled internally using
    the battery's nominal power.

    Attributes:
        observation_space: Box space of shape (12,) with element-wise bounds in [-1.0, 1.0] for
            time features, and [0.0, 1.0] for SOC and normalized production/consumption/prices,
            and 'is_holiday' flag.
        action_space: Box space of shape (1,) representing normalized net power in [-1.0, 1.0].
            Positive values request charging, negative values request discharging.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        csv_path: str,
        battery_config: dict[str, float] | None = None,
        delta_t: float = config.DEFAULT_DELTA_T,
        usage_price: float = config.DEFAULT_REWARD_CONFIG['usage_price'],
        violation_penalty: float = config.DEFAULT_REWARD_CONFIG['violation_penalty'],
    ) -> None:
        """Initialize the energy storage environment.

        Args:
            csv_path: Path to CSV file with columns: datetime, production, consumption,
                buy_price, sell_price.
            battery_config: Dictionary with battery parameters. Defaults to DEFAULT_BATTERY_CONFIG.
            delta_t: Time step duration in hours. Defaults to 1.0.
            usage_price: Price for energy usage (penalty for battery losses) in €/kWh.
                Defaults to 0.001.
            violation_penalty: Penalty weight for power constraint violations.
                Defaults to 0.01.

        Raises:
            ValueError: If battery parameters or time step are invalid.

        Note:
            Action format: normalized net power in [-1, 1].
            Denormalized using p_nom = capacity / delta_t.
        """
        self.csv_path = csv_path
        self.delta_t = delta_t
        self.usage_price = usage_price
        self.violation_penalty = violation_penalty

        if battery_config is None:
            battery_config = config.DEFAULT_BATTERY_CONFIG.copy()
        self.battery_config = battery_config

        # extract and validate battery parameters
        self.capacity = battery_config['capacity']
        self.eta_cha = battery_config['efficiency_charge']
        self.eta_dis = battery_config['efficiency_discharge']
        self.p_cha_max = battery_config['power_charge_max']
        self.p_dis_max = battery_config['power_discharge_max']
        self.dod_max = battery_config['dod_max'] / 100.0
        self.soc_min = 1.0 - self.dod_max
        self.soc_max = 1.0

        # nominal power: capacity per time step
        self.p_nom = self.capacity / delta_t

        # load data
        self.data = utils.load_data(csv_path)
        self.time_horizon = len(self.data)

        # initialize helpers
        self.obs_builder = ObservationBuilder(self.data)
        self.physics = BatteryPhysics(
            capacity=self.capacity,
            eta_cha=self.eta_cha,
            eta_dis=self.eta_dis,
            p_cha_max=self.p_cha_max,
            p_dis_max=self.p_dis_max,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            delta_t=self.delta_t,
        )
        self.reward_calc = RewardCalculator(
            usage_price=usage_price,
            violation_penalty=violation_penalty,
            delta_t=self.delta_t,
            eta_cha=self.eta_cha,
            eta_dis=self.eta_dis,
        )

        # observation space
        obs_low = np.array([
            -1.0, -1.0,  # month_sin, month_cos
            -1.0, -1.0,  # day_sin, day_cos
            -1.0, -1.0,  # hour_sin, hour_cos
            0.0,         # soc
            0.0,         # production (normalized)
            0.0,         # consumption (normalized)
            0.0,         # buy_price (normalized)
            0.0,         # sell_price (normalized)
            0.0,         # is_holiday
        ], dtype=np.float32)
        obs_high = np.array([
            1.0, 1.0,    # month_sin, month_cos
            1.0, 1.0,    # day_sin, day_cos
            1.0, 1.0,    # hour_sin, hour_cos
            1.0,         # soc
            1.0,         # production (normalized)
            1.0,         # consumption (normalized)
            1.0,         # buy_price (normalized)
            1.0,         # sell_price (normalized)
            1.0,         # is_holiday
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # action space: normalized net power in [-1, 1]
        # positive = charge, negative = discharge
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # environment state
        self.current_timestep = 0
        self.soc = config.INITIAL_SOC
        self.episode_reward = 0.0

    def _get_obs(self) -> np.ndarray:
        """Get observation for current timestep.

        Returns:
            Observation vector of shape (12,).
        """
        if self.current_timestep >= self.time_horizon:
            return np.zeros(12, dtype=np.float32)
        return self.obs_builder.build_observation(self.current_timestep, self.soc)

    def _get_info(
        self,
        p_cha: float = 0.0,
        p_dis: float = 0.0,
        energy_export: float = 0.0,
        energy_import: float = 0.0,
        production: float = 0.0,
        consumption: float = 0.0,
        sell_price: float = 0.0,
        buy_price: float = 0.0,
        violation: float = 0.0,
    ) -> dict[str, Any]:
        """Build info dictionary for current step.

        Args:
            p_cha: Charging power in kW.
            p_dis: Discharging power in kW.
            energy_export: Energy exported in kWh.
            energy_import: Energy imported in kWh.
            production: Production at current step in kW.
            consumption: Consumption at current step in kW.
            sell_price: Selling price in €/kWh.
            buy_price: Buying price in €/kWh.
            violation: Constraint violation magnitude in kW.

        Returns:
            Dictionary with step information.
        """
        usage_cost = self.usage_price * self.delta_t * (
            self.eta_cha * p_cha - (1 / self.eta_dis) * p_dis
        )
        return {
            'timestep': self.current_timestep,
            'soc': self.soc,
            'episode_reward': self.episode_reward,
            'p_cha': p_cha,
            'p_dis': p_dis,
            'energy_export': energy_export,
            'energy_import': energy_import,
            'production': production,
            'consumption': consumption,
            'sell_price': sell_price,
            'buy_price': buy_price,
            'profit': sell_price * energy_export - buy_price * energy_import - usage_cost,
            'usage_cost': usage_cost,
            'violation': violation,
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed (for reproducibility, not used here).
            options: Additional options (unused).

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        self.current_timestep = 0
        self.soc = config.INITIAL_SOC
        self.episode_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray | float
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step of the environment.

        Args:
            action: Normalized net power in [-1.0, 1.0]. Positive = charge, negative = discharge.
                Shape (1,) or scalar. Internally scaled to [-p_nom, p_nom] and clipped to feasible bounds.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # extract scalar action
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        # scale normalized action to actual power range
        scaled_action = action * self.p_nom

        # current step data
        row = self.data.iloc[self.current_timestep]
        production = float(row['production'])
        consumption = float(row['consumption'])
        sell_price = float(row['sell_price'])
        buy_price = float(row['buy_price'])

        # compute feasible action and violation magnitude
        feasible_action, violation = self.physics.compute_feasible_action(
            scaled_action, production, self.soc
        )

        # compute power direction and energy flows using feasible action
        p_cha, p_dis, energy_export, energy_import = self.physics.compute_power_direction(
            feasible_action, production, consumption
        )

        # update SOC with feasible action
        self.soc = self.physics.update_state_of_charge(self.soc, p_cha, p_dis)

        # compute reward with violation penalty
        reward = self.reward_calc.compute_reward(
            p_cha, p_dis, energy_export, energy_import, sell_price, buy_price, violation
        )
        self.episode_reward += reward

        # store timestep before advancing
        step_timestep = self.current_timestep

        # advance timestep
        self.current_timestep += 1
        terminated = self.current_timestep >= self.time_horizon
        truncated = False

        # get observation for next state
        obs = self._get_obs()

        # build info dict with step data
        info = self._get_info(
            p_cha=p_cha,
            p_dis=p_dis,
            energy_export=energy_export,
            energy_import=energy_import,
            production=production,
            consumption=consumption,
            sell_price=sell_price,
            buy_price=buy_price,
            violation=violation,
        )
        info['timestep'] = step_timestep

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def milp_solution(self) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute optimal battery actions using MILP solver.

        Solves the energy arbitrage problem to generate optimal
        charging/discharging actions for the entire episode.

        Returns:
            Tuple of:
                - np.ndarray: Optimal normalized net power actions of shape (T,) in [-1, 1].
                - pd.DataFrame: Detailed MILP solution with soc, power_charge,
                    power_discharge, energy_export, energy_import, net_profit.

        Raises:
            ValueError: If the MILP problem is unbounded or infeasible.
        """
        try:
            df_solution = milp_solver(
                csv_path=self.csv_path,
                delta_t=self.delta_t,
                battery_config=self.battery_config,
                usage_price=self.usage_price,
                initial_soc=config.INITIAL_SOC
            )

            # compute net power actions and normalize to [-1, 1]
            net_power = df_solution['power_charge'].values - df_solution['power_discharge'].values
            optimal_actions = net_power / self.p_nom

            # clip to valid action range
            optimal_actions = np.clip(optimal_actions, -1.0, 1.0)

            return optimal_actions.astype(np.float32), df_solution

        except ValueError as e:
            raise ValueError(
                f'Failed to compute optimal actions: {str(e)}'
            ) from e