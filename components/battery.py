from typing import Union

import pandas as pd
import numpy as np


class Battery:

    def __init__(self,
                 E_max: Union[float, int],
                 P_max_bat: Union[float, int],
                 SoC_init: float = 0.1,
                 SoC_min: float = 0.1,
                 SoC_max: float = 0.9,
                 ch_eff: float = 0.9,
                 dch_eff: float = 0.9,
                 t_lpf_bat: int = 100
                 ):

        self.capacity = E_max
        self.max_power = P_max_bat
        self.soc_init = SoC_init
        self.soc_min = SoC_min
        self.soc_max = SoC_max
        self.ch_eff = ch_eff
        self.dch_eff = dch_eff
        self.t_lpf = t_lpf_bat

        self.soc = pd.Series()
        self.charging_power = pd.Series()
        self.discharging_power = pd.Series()


    def __call__(self, load: pd.Series, solar: pd.Series):

        time_steps = len(load)

        SoC = np.zeros(time_steps + 1)
        Pch_bat = np.zeros(time_steps)
        Pdch_bat = np.zeros(time_steps)
        Grid_power = np.zeros(time_steps)


        index = load.index
        load = load.values
        solar = solar.values

        SoC[0] = self.soc_init

        p_bat_prev = 0
        p_bat_temp = 0
        p_bat_temp_prev = 0

        for t in range(time_steps):
            if solar[t] > load[t]:
                # Battery charge algorithm
                # Check if battery is fully charged
                if SoC[t] < self.soc_max:
                    # Calculate required power to reach SoC_max in one second
                    p_cap_bat = self.capacity * ((self.soc_max - SoC[t]) / self.ch_eff) * 3600
                    p_bat_temp = min(p_cap_bat, self.max_power, solar[t] - load[t])
                    # Apply LPF
                    Pch_bat[t] = (p_bat_temp + p_bat_temp_prev) / (2 * self.t_lpf + 1) + \
                                 ((2 * self.t_lpf - 1) / (2 * self.t_lpf + 1)) * p_bat_prev

                    p_bat_prev = Pch_bat[t]
                    p_bat_temp_prev = p_bat_temp
                else:
                    # No charge can be applied
                    Pch_bat[t] = 0
                    p_bat_prev = 0
                    p_bat_temp_prev = 0

                # Discharge is disabled
                Pdch_bat[t] = 0
                # Calculate final grid power
                Grid_power[t] = solar[t] - load[t] - Pch_bat[t]
                # New SoC of the battery
                SoC[t + 1] = SoC[t] + Pch_bat[t] * self.ch_eff / (3600 * self.capacity)

            else:
                # Battery discharge algorithm
                # Check if battery is fully discharged
                if SoC[t] > self.soc_min:
                    # Calculate required power to reach SoC_min in one second
                    p_cap_bat = self.capacity * (SoC[t] - self.soc_min) * self.dch_eff * 3600
                    # Select the minimum value among Pmax_bat, the above calculated power, and Load-PV_power
                    p_bat_temp = min(p_cap_bat, self.max_power, load[t] - solar[t])
                    # Apply LPF
                    Pdch_bat[t] = (p_bat_temp + p_bat_temp_prev) / (2 * self.t_lpf + 1) + \
                                  ((2 * self.t_lpf - 1) / (2 * self.t_lpf + 1)) * p_bat_prev

                    p_bat_prev = Pdch_bat[t]
                    p_bat_temp_prev = p_bat_temp
                else:
                    # No discharge can be applied
                    Pdch_bat[t] = 0
                    p_bat_prev = 0
                    p_bat_temp_prev = 0

                # Charge is disabled
                Pch_bat[t] = 0
                # Calculate final grid power
                Grid_power[t] = solar[t] - load[t] + Pdch_bat[t]
                # New SoC of the battery
                SoC[t + 1] = SoC[t] - (Pdch_bat[t] / self.dch_eff) / (3600 * self.capacity)

        self.soc = pd.Series(index=index, data=SoC[1:])
        self.charging_power = pd.Series(index=index, data=1000*Pch_bat)
        self.discharging_power = pd.Series(index=index, data=1000*Pdch_bat)

        grid_power = pd.Series(index=index, data=Grid_power)

        return grid_power
