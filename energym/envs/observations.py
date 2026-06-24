"""Observation computation utilities for energy storage environment."""

import calendar
from typing import Any

import numpy as np
import pandas as pd

from energym import utils


class ObservationBuilder:
    """Build observation vectors with time-based and market-based features."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize observation builder with dataset statistics.

        Args:
            data: DataFrame with columns: datetime, production, consumption, buy_price, sell_price.
        """
        self.data = data
        self._compute_normalization_stats()

    def _compute_normalization_stats(self) -> None:
        """Compute MinMax normalization statistics for numerical features."""
        self.prod_min = float(self.data['production'].min())
        self.prod_max = float(self.data['production'].max())
        self.cons_min = float(self.data['consumption'].min())
        self.cons_max = float(self.data['consumption'].max())
        self.buy_price_min = float(self.data['buy_price'].min())
        self.buy_price_max = float(self.data['buy_price'].max())
        self.sell_price_min = float(self.data['sell_price'].min())
        self.sell_price_max = float(self.data['sell_price'].max())

    def get_normalized_feature(
        self, value: float,
        feature_min: float,
        feature_max: float
    ) -> float:
        """Normalize a single feature using stored statistics.

        Args:
            value: Feature value to normalize.
            feature_min: Minimum value for normalization.
            feature_max: Maximum value for normalization.

        Returns:
            Normalized value in [0, 1].
        """
        if feature_min >= feature_max:
            return 0.0
        normalized = (value - feature_min) / (feature_max - feature_min)
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def get_days_in_month(year: int, month: int) -> int:
        """Get the number of days in a given month, accounting for leap years.

        Args:
            year: Year (e.g., 2025).
            month: Month (1-12).

        Returns:
            Number of days in the month.
        """
        _, days_in_month = calendar.monthrange(year, month)
        return days_in_month

    def build_observation(self, timestep: int, soc: float) -> np.ndarray:
        """Build observation vector at the given timestep.

        Observation contains 12 elements:
            - Time features (6): sin/cos encoded month, day (month-wise), hour
            - SOC (1): current state of charge
            - Numerical features (4): production, consumption, buy_price, sell_price (normalized)
            - Holiday flag (1): whether day is Sunday or Italian holiday

        Args:
            timestep: Timestep index in the dataset.
            soc: Current state of charge.

        Returns:
            Observation vector of shape (12,) with dtype float32.
        """
        row = self.data.iloc[timestep]
        dt = row['datetime']

        # cyclical time encoding
        month_sin, month_cos = utils.cyclical_encode(dt.month, 12)
        days_in_month = self.get_days_in_month(dt.year, dt.month)
        day_sin, day_cos = utils.cyclical_encode(dt.day, days_in_month)
        hour_sin, hour_cos = utils.cyclical_encode(dt.hour, 24)

        # normalized numerical features
        prod_norm = self.get_normalized_feature(
            row['production'], self.prod_min, self.prod_max
        )
        cons_norm = self.get_normalized_feature(
            row['consumption'], self.cons_min, self.cons_max
        )
        buy_price_norm = self.get_normalized_feature(
            row['buy_price'], self.buy_price_min, self.buy_price_max
        )
        sell_price_norm = self.get_normalized_feature(
            row['sell_price'], self.sell_price_min, self.sell_price_max
        )

        # holiday/sunday flag
        is_holiday = float(utils.is_holiday_or_weekend(dt))

        obs = np.array([
            month_sin, month_cos,
            day_sin, day_cos,
            hour_sin, hour_cos,
            soc,
            prod_norm, cons_norm, buy_price_norm, sell_price_norm,
            is_holiday,
        ], dtype=np.float32)

        return obs
