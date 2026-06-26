"""Physics-based battery dynamics and power balance calculations."""

import numpy as np


class BatteryPhysics:
    """Handle battery state dynamics and power constraint enforcement."""

    def __init__(
        self,
        capacity: float,
        eta_cha: float,
        eta_dis: float,
        p_cha_max: float,
        p_dis_max: float,
        soc_min: float,
        soc_max: float,
        delta_t: float,
    ) -> None:
        """Initialize battery physics parameters.

        Args:
            capacity: Battery capacity in kWh.
            eta_cha: Charging efficiency (0-1).
            eta_dis: Discharging efficiency (0-1).
            p_cha_max: Maximum charging power in kW.
            p_dis_max: Maximum discharging power in kW.
            soc_min: Minimum SOC due to depth of discharge.
            soc_max: Maximum SOC (typically 1.0).
            delta_t: Time step duration in hours.
        """
        self.capacity = capacity
        self.eta_cha = eta_cha
        self.eta_dis = eta_dis
        self.p_cha_max = p_cha_max
        self.p_dis_max = p_dis_max
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.delta_t = delta_t

    def compute_feasible_action(
        self,
        scaled_action: float,
        production: float,
        soc: float
    ) -> tuple[float, float]:
        """Compute feasible net power action respecting all constraints.

        Constraints considered:
        - Maximum chargeable power (p_cha_max) and battery capacity bounds
        - Production available (cannot charge more than production)
        - Maximum dischargeable power (p_dis_max) and battery energy bounds
        - SOC limits due to depth of discharge

        Args:
            scaled_action: Denormalized net power in [-p_nom, p_nom]. Positive for charging,
                negative for discharging.
            production: Production at this timestep in kW.
            soc: Current state of charge.

        Returns:
            Tuple of (feasible_action, violation) where violation quantifies how much
            the original action exceeded feasible bounds.
        """
        # max chargeable power: limited by battery, production, and SOC headroom
        # accounts for charging efficiency: delta_soc = (1/C) * eta_cha * p_cha * dt
        space_to_charge = (self.soc_max - soc) * self.capacity / self.eta_cha / self.delta_t
        p_cha_max_feasible = min(self.p_cha_max, production, space_to_charge)

        # max dischargeable power: limited by battery and SOC floor
        # accounts for discharge efficiency: delta_soc = -(1/C) * (1/eta_dis) * p_dis * dt
        energy_available = (soc - self.soc_min) * self.capacity
        p_dis_max_feasible = min(self.p_dis_max, energy_available * self.eta_dis / self.delta_t)

        # clip action to feasible range
        feasible_action = float(np.clip(
            scaled_action, -p_dis_max_feasible, p_cha_max_feasible
        ))

        # compute violation: magnitude by which original action exceeded bounds
        if scaled_action > p_cha_max_feasible:
            violation = float(scaled_action - p_cha_max_feasible)
        elif scaled_action < -p_dis_max_feasible:
            violation = float(-scaled_action - p_dis_max_feasible)
        else:
            violation = 0.0

        return feasible_action, violation

    def compute_power_direction(
        self,
        action: float,
        production: float,
        consumption: float
    ) -> tuple[float, float, float, float]:
        """Compute charging/discharging power and energy export/import from net action.

        The power balance equation is:
            energy_export/dt - energy_import/dt - production + consumption - p_dis + p_cha = 0

        Given a net action (positive for charging, negative for discharging):
            - p_cha = max(0, action)
            - p_dis = max(0, -action)

        Args:
            action: Net power action (positive for charging, negative for discharging) in kW.
            production: Production at this timestep in kW.
            consumption: Consumption at this timestep in kW.

        Returns:
            Tuple of (p_cha, p_dis, energy_export, energy_import).
        """
        p_cha = max(0.0, action)
        p_dis = max(0.0, -action)

        # power balance: energy_export/dt - energy_import/dt =
        #   = production - consumption + p_dis - p_cha
        net_power = production - consumption + p_dis - p_cha

        if net_power >= 0:
            energy_export = net_power * self.delta_t
            energy_import = 0.0
        else:
            energy_export = 0.0
            energy_import = -net_power * self.delta_t

        return p_cha, p_dis, energy_export, energy_import

    def update_state_of_charge(self, soc: float, p_cha: float, p_dis: float) -> float:
        """Update battery state of charge using physics-based dynamics.

        SOC dynamics:
            s[t+1] = s[t] + (1/C_nom) * (eta_cha * p_cha - (1/eta_dis) * p_dis) * delta_t

        Args:
            soc: Current state of charge.
            p_cha: Charging power in kW.
            p_dis: Discharging power in kW.

        Returns:
            Updated state of charge, clipped to [soc_min, soc_max].
        """
        delta_soc = (1.0 / self.capacity) * (
            self.eta_cha * p_cha - (1.0 / self.eta_dis) * p_dis
        ) * self.delta_t
        new_soc = soc + delta_soc
        return float(np.clip(new_soc, self.soc_min, self.soc_max))
