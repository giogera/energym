"""Default configuration for EnergyStorageEnv."""

DEFAULT_BATTERY_CONFIG = {
    'capacity': 5.12,
    'efficiency_charge': 0.95,
    'efficiency_discharge': 0.95,
    'power_charge_max': 2.5,
    'power_discharge_max': 2.5,
    'dod_max': 90.0,
}

DEFAULT_REWARD_CONFIG = {
    'usage_price': 0.001,
    'violation_penalty': 0.01,
}

DEFAULT_DELTA_T = 1.0

INITIAL_SOC = 0.5
