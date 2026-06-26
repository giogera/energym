"""MILP solver for computing optimal battery control actions."""

import pandas as pd
import numpy as np
import pulp
import time


def milp_solver(
    csv_path: str,
    delta_t: float,
    battery_config: dict[str, float],
    usage_price: float = 0.001,
    initial_soc: float = 0.5
) -> pd.DataFrame:
    """Solve mixed integer linear program to maximize net energy profit.

    Computes optimal battery charging/discharging actions over a time horizon
    to maximize energy arbitrage profit subject to battery constraints.

    Args:
        csv_path: Path to CSV file with columns: datetime, production,
            consumption, buy_price, sell_price.
        delta_t: Time step duration in hours.
        battery_config: Dictionary with keys:
            - capacity: Battery capacity in kWh.
            - efficiency_charge: Charging efficiency (0 < eta <= 1).
            - efficiency_discharge: Discharging efficiency (0 < eta <= 1).
            - power_charge_max: Maximum charging power in kW.
            - power_discharge_max: Maximum discharging power in kW.
            - dod_max: Maximum depth of discharge in percentage (0 < DoD <= 100).
        usage_price: Price for energy usage in €/kWh. Defaults to 0.001.
        initial_soc: Initial state of charge (0 <= initial_soc <= 1). Defaults to 0.5.

    Returns:
        DataFrame with optimal actions: datetime, production, consumption,
        buy_price, sell_price, soc, power_charge, power_discharge,
        energy_export, energy_import, net_profit.

    Raises:
        ValueError: If the optimization problem is unbounded or infeasible.
    """
    # load time series data
    data = pd.read_csv(csv_path, parse_dates=['datetime'])
    E = data['production'].values
    I = data['consumption'].values
    pi_s = data['sell_price'].values
    pi_b = data['buy_price'].values

    # battery parameters
    C_nom = battery_config['capacity']
    eta_cha = battery_config['efficiency_charge']
    eta_dis = battery_config['efficiency_discharge']
    P_max_cha = battery_config['power_charge_max']
    P_max_dis = battery_config['power_discharge_max']
    DoD_max = battery_config['dod_max'] / 100.0
    S_ini = initial_soc
    S_end = S_ini
    pi_w = usage_price

    # time horizon
    T = len(data)

    # define optimization problem
    model = pulp.LpProblem('Battery_MILP', pulp.LpMaximize)

    # decision variables
    p_cha = [pulp.LpVariable(f'p_cha_{t}', lowBound=0, upBound=P_max_cha) for t in range(T)]
    p_dis = [pulp.LpVariable(f'p_dis_{t}', lowBound=0, upBound=P_max_dis) for t in range(T)]
    e_gri = [pulp.LpVariable(f'e_gri_{t}', lowBound=0) for t in range(T)]
    i_gri = [pulp.LpVariable(f'i_gri_{t}', lowBound=0) for t in range(T)]
    delta = [pulp.LpVariable(f'delta_{t}', cat='Binary') for t in range(T)]
    s = [pulp.LpVariable(f's_{t}', lowBound=1 - DoD_max, upBound=1.0) for t in range(T + 1)]

    # objective: maximize energy arbitrage profit
    model += pulp.lpSum(
        pi_s[t] * e_gri[t] - pi_b[t] * i_gri[t]
        - pi_w * delta_t * (eta_cha * p_cha[t] - (1 / eta_dis) * p_dis[t])
        for t in range(T)
    )

    # constraints
    model += s[0] == S_ini
    model += s[T] == S_end

    for t in range(T):
        # state-of-charge dynamics
        model += (
            s[t + 1] == s[t] + (1 / C_nom)
            * (eta_cha * p_cha[t] - (1 / eta_dis) * p_dis[t]) * delta_t
        )

        # charging limited by production
        model += p_cha[t] <= E[t]

        # complementarity: cannot charge and discharge simultaneously
        model += e_gri[t] <= E[t] * delta_t * delta[t]
        model += i_gri[t] <= I[t] * delta_t * (1 - delta[t])

        # power balance
        model += (
            e_gri[t] / delta_t - i_gri[t] / delta_t - E[t] + I[t]
            - p_dis[t] + p_cha[t] == 0
        )

    # solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    start = time.perf_counter()
    status = model.solve(solver)
    elapsed = time.perf_counter() - start

    # extract results
    status_str = pulp.LpStatus[status]
    if status_str == 'Optimal':
        print(f'Optimal solution found in {elapsed:.2f} seconds')
        df = pd.DataFrame({
            'datetime': data['datetime'],
            'production': E,
            'consumption': I,
            'buy_price': pi_b,
            'sell_price': pi_s,
            'soc': [pulp.value(s[t]) for t in range(T)],
            'power_charge': [pulp.value(p_cha[t]) for t in range(T)],
            'power_discharge': [pulp.value(p_dis[t]) for t in range(T)],
            'energy_export': [pulp.value(e_gri[t]) for t in range(T)],
            'energy_import': [pulp.value(i_gri[t]) for t in range(T)],
        })

        # compute step-wise profit
        df['net_profit'] = (
            df['sell_price'] * df['energy_export']
            - df['buy_price'] * df['energy_import']
            - pi_w * delta_t * (eta_cha * df['power_charge'] - (1 / eta_dis) * df['power_discharge'])
        )

        print(f"Total net profit: {df['net_profit'].sum():.4f} €")
        return df
    else:
        raise ValueError(f'MILP solver failed: {status_str}. Problem may be unbounded or infeasible.')
