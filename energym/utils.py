"""Utility functions for the energy environment."""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import holidays


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate energy time series data from CSV.

    Args:
        csv_path: Path to CSV file with columns: datetime, production, consumption,
            buy_price, sell_price.

    Returns:
        DataFrame with parsed datetime and validated columns.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=['datetime'])
    except FileNotFoundError:
        raise FileNotFoundError(f'CSV file not found at {csv_path}')

    required_cols = ['datetime', 'production', 'consumption', 'buy_price', 'sell_price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')

    if df.empty:
        raise ValueError('CSV file is empty')

    return df.reset_index(drop=True)


def minmax_normalize(
    values: np.ndarray,
    feature_min: float | None = None,
    feature_max: float | None = None
) -> tuple[np.ndarray, float, float]:
    """Normalize values to [0, 1] using MinMax scaling.

    Args:
        values: Array of values to normalize.
        feature_min: Minimum value for normalization. If None, use min of values.
        feature_max: Maximum value for normalization. If None, use max of values.

    Returns:
        Tuple of (normalized_values, min_used, max_used).

    Raises:
        ValueError: If feature_min >= feature_max.
    """
    if feature_min is None:
        feature_min = float(np.min(values))
    if feature_max is None:
        feature_max = float(np.max(values))

    if feature_min >= feature_max:
        raise ValueError(f'feature_min ({feature_min}) must be < feature_max ({feature_max})')

    normalized = (values - feature_min) / (feature_max - feature_min)
    return np.clip(normalized, 0.0, 1.0), feature_min, feature_max


def cyclical_encode(value: int, period: int) -> tuple[float, float]:
    """Encode a cyclic value (e.g., month, hour) as sin/cos pair.

    Args:
        value: Value to encode (e.g., month 1-12, hour 0-23).
        period: Period of the cycle (e.g., 12 for months, 24 for hours).

    Returns:
        Tuple of (sin_encoded, cos_encoded).
    """
    angle = 2 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def is_holiday_or_weekend(date: datetime) -> bool:
    """Check if date is a Sunday or a national holiday (Italy).

    Args:
        date: Datetime object to check.

    Returns:
        True if the date is a Sunday or Italian holiday, False otherwise.
    """
    is_sunday = date.weekday() == 6
    italy_holidays = holidays.Italy()
    is_italian_holiday = date.date() in italy_holidays

    return is_sunday or is_italian_holiday
