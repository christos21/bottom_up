from typing import Union

import pandas as pd


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

