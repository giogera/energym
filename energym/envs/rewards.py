"""Class for computing step reward combining profit and constraint violation penalties."""


class RewardCalculator:
    """Compute step reward combining profit and constraint violation penalties."""

    def __init__(
        self,
        usage_price: float,
        violation_penalty: float,
        delta_t: float,
        eta_cha: float,
        eta_dis: float,
    ) -> None:
        """Initialize reward calculator.

        Args:
            usage_price: Price for energy usage (penalty for battery losses) in €/kWh.
            violation_penalty: Penalty weight for power constraint violations.
            delta_t: Time step duration in hours.
            eta_cha: Charging efficiency (0-1).
            eta_dis: Discharging efficiency (0-1).
        """
        self.usage_price = usage_price
        self.violation_penalty = violation_penalty
        self.delta_t = delta_t
        self.eta_cha = eta_cha
        self.eta_dis = eta_dis

    def compute_reward(
        self,
        p_cha: float,
        p_dis: float,
        energy_export: float,
        energy_import: float,
        sell_price: float,
        buy_price: float,
        violation: float,
    ) -> float:
        """Compute step reward.

        Reward = net_profit - violation_penalty

        where:
            net_profit = sell_price * energy_export - buy_price * energy_import
                         - usage_price * delta_t * (eta_cha * p_cha - (1/eta_dis) * p_dis)

            violation_penalty = violation_penalty * violation^2

        Args:
            p_cha: Charging power in kW.
            p_dis: Discharging power in kW.
            energy_export: Energy exported to grid in kWh.
            energy_import: Energy imported from grid in kWh.
            sell_price: Current selling price in €/kWh.
            buy_price: Current buying price in €/kWh.
            violation: Constraint violation magnitude from action clipping in kW.

        Returns:
            Reward value in €.
        """
        # net profit 
        net_profit = (
            sell_price * energy_export
            - buy_price * energy_import
            - self.usage_price * self.delta_t * (
                self.eta_cha * p_cha - (1 / self.eta_dis) * p_dis
            )
        )

        # constraint violation penalty
        penalty = self.violation_penalty * (violation ** 2)

        return float(net_profit - penalty)
