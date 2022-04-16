import os
from typing import Union, Tuple, Dict, List

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_config
import indexes
from utils import day_type, natural_keys, phase_allocation, PHASES

from components.solar import PV
from components.battery import Battery


class SmartHome:
    """
    Class representing a smart home with solar panels and battery.
    """

    def __init__(self, bus_name: str, home_info: pd.Series):
        """
        Initialization of a smart home object.
        :param bus_name: str
        :param home_info: pd.Series
        """
        self.name = bus_name
        self.number_of_appliances = int(home_info['appliances'])
        self.appliances = [appliance for appliance in data_config.APPLIANCES if home_info[appliance]]

        self.single_phase, self.selected_phase, self.appliance_to_phase = self.phase_allocation(home_info)

        self.appliance_profile = {key: int(home_info[key + '_profile']) for key in data_config.APPLIANCES}
        self.appliance_profile['AlwaysOn'] = int(home_info['AlwaysOn_profile'])

        self.has_battery, self.battery = self.deduce_battery(home_info)
        self.has_pv, self.pv = self.deduce_solar(home_info)

        self.p_cap = home_info['p_cap']

        self.P = pd.DataFrame()
        self.Q = pd.DataFrame()
        self.PV = pd.DataFrame()
        self.grid_power = pd.DataFrame()
        self.total_grid_power = pd.Series()

        self.indexes = {}
        self.simulation_appliances = []
        self.simulation_appliance_to_phase = {}

    def phase_allocation(
            self,
            home_info: pd.Series
    ) -> Tuple[bool, Union[None, str], Dict[str, str]]:
        """
        Perform phase allocation for the appliances.
        If home is single phase, then it is connected either to a phase specified in home_info or it is
        automatically connected to phase a.
        If the home is three-phased then any electric vehicle will be considered as three-phase as well,
        and the rest of the appliances will be allocated according to phase_allocation() function
        :param home_info: pd.Series
        :return: Tuple[bool, Union[None, str], Dict[str: str]]
        """
        selected_phase = None
        single_phase = True if home_info['phase_number'] in [1, 2, 3] else False

        if single_phase:
            selected_phase = PHASES[home_info['phase_number'] - 1]
            assert home_info['phase_number'] in [1, 2, 3]
            appliance_to_phase = {app: selected_phase for app in self.appliances}

        else:

            if 'BEV' in self.appliances:
                apps_copy = self.appliances.copy()
                apps_copy.remove('BEV')
                appliance_to_phase = phase_allocation(apps_copy)
                appliance_to_phase['BEV'] = 'abc'
            else:
                appliance_to_phase = phase_allocation(self.appliances)

        return single_phase, selected_phase, appliance_to_phase

    @staticmethod
    def deduce_battery(home_info: pd.Series) -> Tuple[bool, Union[Battery, None]]:
        """
        Deduce if there is battery for this house and creates the Battery object.
        :param home_info: pd.Series
        :return: bool, Battery|None
        """
        if np.isnan(home_info['E_max']):
            has_battery = False
            battery = None
        else:
            has_battery = True

            battery_params = \
                {
                    key: home_info[key] for key in ['SoC_init', 'P_max_bat', 'E_max',
                                                    'SoC_min', 'SoC_max', 'ch_eff',
                                                    'dch_eff', 't_lpf_bat']
                    if not np.isnan(home_info[key])
                }

            battery = Battery(**battery_params)

        return has_battery, battery

    def deduce_solar(self, home_info: pd.Series) -> Tuple[bool, Union[PV, None]]:
        """
        Deduce if there is PV for this house and creates the PV object.
        :param home_info: pd.DataFrame
        :return: bool, PV|None
        """
        pv_rated = None if np.isnan(home_info['PV_rated']) else home_info['PV_rated']
        pv_profile = 0 if (
                            'PV_profile' not in home_info.index or
                            np.isnan(home_info['PV_profile'])
                          ) else home_info['PV_profile']

        if pv_rated:
            has_pv = True
            pv = PV(bus=self.name, pv_rated=pv_rated, single_phase=self.single_phase,
                    selected_phase=self.selected_phase, profile=pv_profile)
        else:
            has_pv = False
            pv = None

        return has_pv, pv

    def set_single_day_aggregated_profile(self, profile_option: int, day: str):
        """
        Creates the load profile for a single day.
        :param profile_option: int
        :param day: str
        :return:
        """
        # initialize active and reactive power dataframes with 0s
        p = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

        q = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

        # Set number of appliances, type of appliances and phase allocation according to the profile_option
        # as mentioned in the manual
        if profile_option == 0:
            # For profile_option = 0, both number_of_appliances and type of appliances are random.
            number_of_appliances = np.random.randint(data_config.NUM_APPLIANCES)
            appliances = random.sample(data_config.APPLIANCES, number_of_appliances)
            # phase allocation
            if not self.single_phase:
                appliance_to_phase = phase_allocation(appliances)
            else:
                appliance_to_phase = {app: self.selected_phase for app in appliances}

        elif profile_option == 1:
            # For profile_option = 1, number_of_appliances is fixed (set at initialization)
            # and type of appliances is random.
            appliances = random.sample(data_config.APPLIANCES, self.number_of_appliances)
            # phase allocation
            if not self.single_phase:
                appliance_to_phase = phase_allocation(appliances)
            else:
                appliance_to_phase = {app: self.selected_phase for app in appliances}

        else:
            # In any other case, both number_of_appliances and type of appliances are fixed and set at initialization.
            appliances = self.appliances
            appliance_to_phase = self.appliance_to_phase

        # The always on load is considered independently for each home.
        appliances += ['AlwaysOn']

        self.simulation_appliances = appliances
        self.simulation_appliance_to_phase = appliance_to_phase

        # Add active and reactive power of each appliance
        for appliance in appliances:
            # get available csv files for the profiles of each appliance
            path_to_profiles = os.path.join(data_config.PROFILES_PATH, appliance, day_type[day])
            available_profiles = os.listdir(path_to_profiles)
            available_profiles.sort(key=natural_keys)

            # For profile option 0, 1 or 2, select a random profile
            if profile_option < 3:
                selected_profile = np.random.choice(available_profiles)
                selected_profile = os.path.join(path_to_profiles, selected_profile)
            else:
                # For profile option 3, assert that the wanted profile exists
                assert self.appliance_profile[appliance] < len(available_profiles), 'No profile {} for {}'.format(
                        self.appliance_profile[appliance], appliance)

                selected_profile = os.path.join(path_to_profiles, available_profiles[self.appliance_profile[appliance]])

            # read the time-series from the csv file
            appliance_power = pd.read_csv(selected_profile, index_col=0)

            # Depending on the appliance type, add the appliance power to the total power.
            if appliance == 'AlwaysOn':
                # in case of always_on, if the home is three-phase both active and reactive power are split
                # between the phases
                if self.single_phase:
                    p[self.selected_phase] += appliance_power.P.values
                    if 'Q' in appliance_power.columns:
                        q[self.selected_phase] += appliance_power.Q.values
                else:
                    for phase in PHASES.values():
                        p[phase] += appliance_power.P.values/3
                        if 'Q' in appliance_power.columns:
                            q[phase] += appliance_power.Q.values/3
                continue

            # in case of electric vehicle, if the home is three-phase both active and reactive power are split
            # between the phases
            elif appliance in ['BEV'] and not self.single_phase:
                for phase in PHASES.values():
                    p[phase] += appliance_power.P.values / 3
                    if 'Q' in appliance_power.columns:
                        q[phase] += appliance_power.Q.values / 3
                continue

            # for any other appliance, both active and reactive power are added to the corresponding phase
            p[appliance_to_phase[appliance]] += appliance_power.P.values
            if 'Q' in appliance_power.columns:
                q[appliance_to_phase[appliance]] += appliance_power.Q.values

        self.P = p
        self.Q = q

    def set_multiple_days_aggregated_profile(self, days: List[str]):
        """
        Creates the load profile for multiple days.
        :param days: [str]
        :return:
        """
        # initialize active and reactive power dataframes with 0s
        p = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days)*60*60*24))

        q = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days)*60*60*24))

        # In the case of multiple days, both the number and the type of appliances are fixed
        # but the profiles are random.
        appliances = self.appliances + ['AlwaysOn']
        appliance_to_phase = self.appliance_to_phase

        self.simulation_appliances = appliances
        self.simulation_appliance_to_phase = appliance_to_phase

        # Iterate for every day
        for k, day in enumerate(days):
            # For each day, iterate through all available appliances
            for appliance in appliances:
                # Get a random profile from the available
                path_to_profiles = os.path.join(data_config.PROFILES_PATH, appliance, day_type[day])
                available_profiles = os.listdir(path_to_profiles)
                available_profiles.sort(key=natural_keys)

                selected_profile = np.random.choice(available_profiles)
                selected_profile = os.path.join(path_to_profiles, selected_profile)

                # Get the appliance power as a dataframe from the csv file
                appliance_power = pd.read_csv(selected_profile, index_col=0)

                # Create the start and end index of each day
                start_index = str(k) + ' days'
                end_index = start_index + ' 23:59:59'

                # In case of always_on, if the home is three-phase both active and reactive power are split
                # between the phases
                if appliance == 'AlwaysOn':
                    if self.single_phase:
                        p.loc[start_index:end_index, self.selected_phase] += \
                            appliance_power.P.values
                        if 'Q' in appliance_power.columns:
                            q.loc[start_index:end_index, self.selected_phase] += \
                                appliance_power.Q.values
                    else:
                        for phase in PHASES.values():
                            p.loc[start_index:end_index, phase] += appliance_power.P.values/3
                            if 'Q' in appliance_power.columns:
                                q.loc[start_index:end_index, phase] += appliance_power.Q.values/3
                    continue

                # in case of electric vehicle, if the home is three-phase both active and reactive power are split
                # between the phases
                elif appliance == 'BEV' and not self.single_phase:
                    for phase in PHASES.values():
                        p.loc[start_index:end_index, phase] += appliance_power.P.values/3
                        if 'Q' in appliance_power.columns:
                            q.loc[start_index:end_index, phase] += appliance_power.Q.values/3
                    continue

                # for any other appliance, both active and reactive power are added to the corresponding phase
                p.loc[start_index:end_index, appliance_to_phase[appliance]] += appliance_power.P.values
                if 'Q' in appliance_power.columns:
                    q.loc[start_index:end_index, appliance_to_phase[appliance]] += appliance_power.Q.values

        self.P = p
        self.Q = q

    def set_load_from_arrays(
            self,
            p_array: np.ndarray,
            q_array: np.ndarray,
            days: List[str],
            phase: Union[str, None] = None):
        """
        Creates the load profile from arrays.
        :param p_array: np.array
        :param q_array: np.array
        :param days: [str]
        :param phase: None | str
        :return:
        """

        if phase:
            # initialize active and reactive power dataframes with 0s
            p = pd.DataFrame(0, columns=list(PHASES.values()),
                             index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

            q = pd.DataFrame(0, columns=list(PHASES.values()),
                             index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

            p[phase] = p_array
            q[phase] = q_array
        else:
            # initialize active and reactive power dataframes with 0s
            p = pd.DataFrame(p_array, columns=list(PHASES.values()),
                             index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

            q = pd.DataFrame(q_array, columns=list(PHASES.values()),
                             index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

        self.P = p.copy()
        self.Q = q.copy()

    def set_pv(
            self,
            days: List[str],
            month: int,
            from_array: bool = False,
            pv_array: Union[np.ndarray, None] = None):
        """
        Creates the production profile for the solar panels.
        :param days: [str]
        :param month: int
        :param from_array: bool
        :param pv_array: np.array
        :return:
        """
        if self.has_pv:
            self.pv.set_power(days, month, from_array, pv_array)
            self.PV = self.pv.P.sum(axis=1)
        else:
            self.PV = pd.Series(index=self.P.index, data=0)

    def set_battery(self):
        """
        Creates the battery charging and discharging active power time-series.
        Both self.P and self.PV have to be initialized before calling this method.
        :return:
        """
        # get the total active power from all phases
        p = self.P.sum(axis=1)

        # If the home has a battery, apply the battery function
        if self.has_battery:
            self.total_grid_power = 1000 * self.battery(p/1000, self.PV/1000)

        else:
            self.total_grid_power = self.PV - p

    def set_pv_and_battery(
            self,
            days: List[str],
            month: int,
            pv_from_array: bool = False,
            pv_array: Union[np.ndarray, None] = None):
        """
        Method to set both PV and battery. It also calculates the grid power for each phase.
        :param days: [str]
        :param month: int
        :param pv_from_array: bool
        :param pv_array: np.array
        :return:
        """
        self.set_pv(days, month, pv_from_array, pv_array)
        self.set_battery()
        self.calculate_grid_power()

    def reset_pv_and_battery_values(self, df: pd.Series):
        """
        Method that resets PV and battery properties. The time-series for PV production and
        battery charging/discharging are set to None.
        A new dataframe is needed including the properties of the new PV and battery.
        :param df: pd.Series
        :return:
        """

        self.has_battery, self.battery = self.deduce_battery(df)
        self.has_pv, self.pv = self.deduce_solar(df)

        self.PV = pd.Series()
        self.grid_power = pd.Series()
        self.total_grid_power = pd.Series()

    def calculate_grid_power(self):
        """
        Calculates the grid power for each phase.
        :return:
        """
        temp_pv = pd.DataFrame(index=self.P.index, columns=self.P.columns, data=0)
        temp_bat_ch = pd.DataFrame(index=self.P.index, columns=self.P.columns, data=0)
        temp_bat_dch = pd.DataFrame(index=self.P.index, columns=self.P.columns, data=0)
        # In case of single phase home both PV and battery operate in the selected phase.
        if self.single_phase:
            temp_pv[self.selected_phase] = self.PV.values
            if self.has_battery:
                temp_bat_ch[self.selected_phase] = self.battery.charging_power.values
                temp_bat_dch[self.selected_phase] = self.battery.discharging_power.values
        else:
            # In case of three-phase homes, both PV production and battery charging/discharging
            # are split between the phases.
            for phase in temp_pv.columns:
                temp_pv[phase] = self.PV.values/3
                if self.has_battery:
                    temp_bat_ch[phase] = self.battery.charging_power.values/3
                    temp_bat_dch[phase] = self.battery.discharging_power.values/3

        # the grid power for each phase is calculated as all the produced power (PV + discharging) minus
        # the consumed power (loads + charging)
        self.grid_power = temp_pv - self.P - temp_bat_ch + temp_bat_dch

    def mean_auto_consumption_rate(self, battery: bool = True):
        """
        Returns mean auto consumption rate. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: pd.Series
        """
        if battery and self.has_battery:
            r = indexes.mean_auto_consumption_rate(gen=self.PV,  # + self.battery['p_dch_bat'],
                                                   injected=self.total_grid_power.clip(0))
        else:
            injected_power = self.PV - self.P.sum(axis=1)
            r = indexes.mean_auto_consumption_rate(gen=self.PV, injected=injected_power.clip(0))

        return r

    def cover_factors(self, battery: bool = True):
        """
        Returns cover factors. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: (float, float)
        """
        if battery and self.has_battery:
            gamma_s, gamma_d = indexes.cover_factors(gen=self.PV + self.battery.discharging_power,
                                                     load=self.P.sum(axis=1) + self.battery.charging_power)
        else:
            gamma_s, gamma_d = indexes.cover_factors(gen=self.PV, load=self.P.sum(axis=1))

        return gamma_s, gamma_d

    def load_match_index(self, interval: str = '1h', battery: bool = True):
        """
        Returns load match index. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        Parameter 'interval' is used for down-sampling.
        :param interval: str
        :param battery: bool
        :return: pd.Series
        """
        if battery and self.has_battery:
            lmi = indexes.load_match_index(gen=self.PV + self.battery.discharging_power,
                                           load=self.P.sum(axis=1) + self.battery.charging_power,
                                           interval=interval)
        else:
            lmi = indexes.load_match_index(gen=self.PV, load=self.P.sum(axis=1), interval=interval)

        return lmi

    def loss_of_load_probability(self, battery: bool = True):
        """
        Returns loss of load probability. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            lolp = indexes.loss_of_load_probability(gen=self.PV + self.battery.discharging_power,
                                                    load=self.P.sum(axis=1) + self.battery.charging_power)
        else:
            lolp = indexes.loss_of_load_probability(gen=self.PV, load=self.P.sum(axis=1))

        return lolp

    def peaks_above_limit(self, p_limit: float = 5000., battery: bool = True):
        """
        Returns peaks above limit. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param p_limit: float | int
        :param battery: bool
        :return: float
        """

        if battery and self.has_battery:
            pal = indexes.peaks_above_limit(p_exchange=self.total_grid_power, p_limit=p_limit)
        else:
            pal = indexes.peaks_above_limit(p_exchange=self.PV - self.P.sum(axis=1), p_limit=p_limit)
        return pal

    def no_grid_interaction_probability(self, period: str = '15min', limit: float = 0.001, battery: bool = True):
        """
        Returns no grid interaction probability. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        The argument 'period' is used for down-sampling.
        The argument 'limit' is the power limit under which there is no interaction with the grid.
        :param period: str
        :param limit: float
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            ngip = indexes.no_grid_interaction_probability(p_exchange=self.total_grid_power,
                                                           period=period, limit=limit)
        else:
            ngip = indexes.no_grid_interaction_probability(p_exchange=self.PV - self.P.sum(axis=1),
                                                           period=period, limit=limit)
        return ngip

    def one_percent_peak_power(self, period: str = '1s', battery: bool = True):
        """
        Returns one percent peak power. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV. The argument 'period' is used in case down-sampling is preferred.
        :param period: str
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            opp = indexes.one_percent_peak_power(p_exchange=self.total_grid_power, period=period)
        else:
            opp = indexes.one_percent_peak_power(p_exchange=self.PV - self.P.sum(axis=1), period=period)
        return opp

    def capacity_factor(self, battery: bool = True):
        """
        Returns capacity factor. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            cf = indexes.capacity_factor(p_exchange=self.total_grid_power, p_cap=self.p_cap)
        else:
            cf = indexes.capacity_factor(p_exchange=self.PV - self.P.sum(axis=1), p_cap=self.p_cap)
        return cf

    def self_consumption_rate(self, battery: bool = True):
        """
        Returns self consumption rate. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            scr = indexes.self_consumption_rate(self.P.sum(axis=1), self.PV, battery_charge=self.battery.charging_power,
                                                battery_discharge=self.battery.discharging_power)
        else:
            scr = indexes.self_consumption_rate(self.P.sum(axis=1), self.PV)

        return scr

    def self_sufficiency_rate(self, battery: bool = True):
        """
        Returns self sufficiency rate. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            ssr = indexes.self_sufficiency_rate(self.P.sum(axis=1), self.PV, battery_charge=self.battery.charging_power,
                                                battery_discharge=self.battery.discharging_power)
        else:
            ssr = indexes.self_sufficiency_rate(self.P.sum(axis=1), self.PV)

        return ssr

    def battery_utilization_index(self):
        """
        Returns battery utilization index for charging and discharging as a tuple.
        If household has no battery, returns (None, None).
        :return: (float, float) | (None, None)
        """
        if self.has_battery:
            bui = indexes.battery_utilization_index(self.battery.charging_power, self.battery.discharging_power,
                                                    self.battery.capacity, depth_of_discharge=max(1-self.battery.soc))
            bui_charge = bui[0]
            bui_discharge = bui[1]
        else:
            bui_charge, bui_discharge = None, None

        return bui_charge, bui_discharge

    def calculate_indexes(self):
        """
        Returns some end-user related indexes for PV-BESS assessment.
        The indexes are calculated for two scenarios: a) only PV and b) PV and BESS.
        :return: dict
        """

        self.indexes['r_auto_without_battery'] = self.mean_auto_consumption_rate(battery=False)
        self.indexes['r_auto_with_battery'] = self.mean_auto_consumption_rate()

        gamma_s, gamma_d = self.cover_factors(battery=False)
        self.indexes['gamma_s_without_battery'] = gamma_s
        self.indexes['gamma_d_without_battery'] = gamma_d

        gamma_s, gamma_d = self.cover_factors()
        self.indexes['gamma_s_with_battery'] = gamma_s
        self.indexes['gamma_d_with_battery'] = gamma_d

        self.indexes['lmi_1d_without_battery'] = self.load_match_index(battery=False, interval='1d')
        self.indexes['lmi_1d_with_battery'] = self.load_match_index(interval='1d')

        self.indexes['lolp_without_battery'] = self.loss_of_load_probability(battery=False)
        self.indexes['lolp_with_battery'] = self.loss_of_load_probability()

        self.indexes['pal_5kW_without_battery'] = self.peaks_above_limit(battery=False, p_limit=5000)
        self.indexes['pal_5kW_with_battery'] = self.peaks_above_limit(p_limit=5000)

        self.indexes['ngip_15min_without_battery'] = self.no_grid_interaction_probability(battery=False,
                                                                                          period='15min', limit=0.001)
        self.indexes['ngip_15min_with_battery'] = self.no_grid_interaction_probability(period='15min', limit=0.001)

        self.indexes['opp_15min_without_battery'] = self.one_percent_peak_power(battery=False, period='15min')
        self.indexes['opp_15min_with_battery'] = self.one_percent_peak_power(period='15min')

        self.indexes['capacity_factor_without_battery'] = self.capacity_factor(battery=False)
        self.indexes['capacity_factor_with_battery'] = self.capacity_factor()

    def plot_grid_power(self, phases=True):
        """
        Plots the total grid power (load - production).
        If 'phases' is true each phase is plotted separately.
        :param phases: boolean
        :return:
        """
        plt.figure()
        if phases:
            self.grid_power.plot()
            plt.title(self.name + ' - Grid power', fontsize=20)
        else:
            self.total_grid_power.plot()
            plt.title(self.name + ' - Total grid power', fontsize=20)
        plt.show()

    def get_total_load(self):
        """
        Returns total active power (including battery charging) and reactive power.
        :return: pd.Series, pd.Series
        """
        p = self.P.sum(axis=1) + self.battery.charging_power
        q = self.Q.sum(axis=1)
        return p, q

    def get_total_production(self):
        """
        Returns power produced by PV and battery.
        :return: pd.Series
        """
        return self.PV + self.battery.discharging_power

    def plot_load(self):
        """
        Plots household total active and reactive power.
        :return:
        """
        plt.figure()
        p, q = self.get_total_load()
        p.plot()
        q.plot()
        plt.title(self.name + ' - Load', fontsize=20)
        plt.legend(['Active power', 'Reactive power'], fontsize=18)
        plt.show()

    def plot_production(self):
        """
        Plots household total production (PV + battery).
        :return:
        """
        prod = self.get_total_production()
        plt.figure()
        prod.plot()
        plt.title(self.name + ' - Production', fontsize=20)
        plt.show()

    def plot_pv(self):
        """
        Plots household PV production.
        :return:
        """
        plt.figure()
        self.PV.plot()
        plt.title(self.name + ' - PV power', fontsize=20)
        plt.show()

    def plot_load_and_production(self):
        """
        Plots household total active power and total production (PV + battery) in a single figure.
        :return:
        """
        p, _ = self.get_total_load()
        plt.figure()
        p.plot()
        prod = self.get_total_production()
        prod.plot()
        plt.title(self.name, fontsize=20)
        plt.legend(['Total consumption', 'Total production'])
        plt.show()

