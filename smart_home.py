import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt


import data_config
import indexes
from utils import day_type, natural_keys, phase_allocation, PHASES, pv_profile_generator, battery_function



class SmartHome:

    def __init__(self, home, home_info):
        """
        Initialization of a smart home object.

        :param home_info: pd.Series
        """
        self.name = home
        self.number_of_appliances = int(home_info['appliances'])
        self.appliances = [appliance for appliance in data_config.APPLIANCES if home_info[appliance]]

        self.single_phase = True if home_info['phase_number'] in [1, 2, 3] else False

        # Perform phase allocation for the appliances.
        # If home is single phase, then it is connected either to a phase specified in home_info or it is
        # automatically connected to phase a
        if self.single_phase:
            assert home_info['phase_number'] in [1, 2, 3]
            self.appliance_to_phase = {app: PHASES[home_info['phase_number']-1] for app in self.appliances}
            self.selected_phase = PHASES[home_info['phase_number']-1]
        else:
            if 'BEV' in self.appliances:
                apps_copy = self.appliances.copy()
                apps_copy.remove('BEV')
                self.appliance_to_phase = phase_allocation(apps_copy)
                self.appliance_to_phase['BEV'] = 'abc'
            else:
                self.appliance_to_phase = phase_allocation(self.appliances)

            self.selected_phase = None

        self.appliance_profile = {key: int(home_info[key + '_profile']) for key in data_config.APPLIANCES}
        self.appliance_profile['AlwaysOn'] = int(home_info['AlwaysOn_profile'])

        self.battery = {
            key.lower(): None if np.isnan(home_info[key]) else home_info[key] for key in
                ['SoC_init', 'P_max_bat', 'E_max', 'SoC_min', 'SoC_max', 'ch_eff', 'dch_eff', 't_lpf_bat']
        }

        if self.battery['p_max_bat']:
            self.has_battery = True
        else:
            self.has_battery = False

        self.battery['soc'] = None
        self.battery['p_ch_bat'] = None
        self.battery['p_dch_bat'] = None

        self.pv_rated = None if np.isnan(home_info['PV_rated']) else home_info['PV_rated']

        if self.pv_rated:
            self.has_pv = True
        else:
            self.has_pv = False

        self.p_cap = home_info['p_cap']

        self.P = None
        self.Q = None
        self.PV = None

        self.grid_power = pd.Series()
        self.total_grid_power = pd.Series()

        self.indexes = {}

        self.simulation_appliances = []
        self.simulation_appliance_to_phase = {}

    def set_single_day_aggregated_profile(self, profile_option, day):

        p = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

        q = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

        if profile_option == 0:
            number_of_appliances = np.random.randint(data_config.NUM_APPLIANCES)
            appliances = random.sample(data_config.APPLIANCES, number_of_appliances)
            if not self.single_phase:
                appliance_to_phase = phase_allocation(appliances)
            else:
                appliance_to_phase = {app: self.selected_phase for app in appliances}
            # and random profile
        elif profile_option == 1:
            appliances = random.sample(data_config.APPLIANCES, self.number_of_appliances)
            if not self.single_phase:
                appliance_to_phase = phase_allocation(appliances)
            else:
                appliance_to_phase = {app: self.selected_phase for app in appliances}
            # and random profile
        else:
            appliances = self.appliances
            appliance_to_phase = self.appliance_to_phase
            # for 2 random profile, for 3 fixed

        appliances += ['AlwaysOn']
        self.simulation_appliances = appliances
        self.simulation_appliance_to_phase = appliance_to_phase

        for appliance in appliances:
            path_to_profiles = os.path.join(data_config.PROFILES_PATH, appliance, day_type[day])
            available_profiles = os.listdir(path_to_profiles)
            available_profiles.sort(key=natural_keys)

            if profile_option < 3:
                selected_profile = np.random.choice(available_profiles)
                selected_profile = os.path.join(path_to_profiles, selected_profile)
            else:
                assert self.appliance_profile[appliance] < len(available_profiles), 'No profile {} for {}'.format(
                        self.appliance_profile[appliance], appliance)

                selected_profile = os.path.join(path_to_profiles, available_profiles[self.appliance_profile[appliance]])

            appliance_power = pd.read_csv(selected_profile, index_col=0)

            if appliance == 'AlwaysOn':
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

            elif appliance == 'BEV' and not self.single_phase:
                for phase in PHASES.values():
                    p[phase] += appliance_power.P.values / 3
                    if 'Q' in appliance_power.columns:
                        q[phase] += appliance_power.Q.values / 3

                continue

            p[appliance_to_phase[appliance]] += appliance_power.P.values

            if 'Q' in appliance_power.columns:
                q[appliance_to_phase[appliance]] += appliance_power.Q.values

        self.P = p
        self.Q = q

    def set_multiple_days_aggregated_profile(self, days):
        p = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days)*60*60*24))

        q = pd.DataFrame(0, columns=list(PHASES.values()),
                         index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days)*60*60*24))

        appliances = self.appliances + ['AlwaysOn']
        appliance_to_phase = self.appliance_to_phase

        self.simulation_appliances = appliances
        self.simulation_appliance_to_phase = appliance_to_phase

        # random profiles
        for k, day in enumerate(days):
            for appliance in appliances:
                path_to_profiles = os.path.join(data_config.PROFILES_PATH, appliance, day_type[day])
                available_profiles = os.listdir(path_to_profiles)
                available_profiles.sort(key=natural_keys)

                selected_profile = np.random.choice(available_profiles)
                selected_profile = os.path.join(path_to_profiles, selected_profile)

                appliance_power = pd.read_csv(selected_profile, index_col=0)

                start_index = str(k) + ' days'
                end_index = start_index + ' 23:59:59'

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

                elif appliance == 'BEV' and not self.single_phase:
                    for phase in PHASES.values():
                        p.loc[start_index:end_index, phase] += appliance_power.P.values/3
                        if 'Q' in appliance_power.columns:
                            q.loc[start_index:end_index, phase] += appliance_power.Q.values/3

                    continue

                p.loc[start_index:end_index, appliance_to_phase[appliance]] += appliance_power.P.values
                if 'Q' in appliance_power.columns:
                    q.loc[start_index:end_index, appliance_to_phase[appliance]] += appliance_power.Q.values

        self.P = p
        self.Q = q

    def set_pv(self, days, month):
        if len(days) == 1:
            if self.has_pv:
                self.PV = pd.Series(data=self.pv_rated*pv_profile_generator(month), name=days[0],
                                    index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))
                self.PV *= 1000
            else:
                self.PV = pd.Series(data=0, name=days[0],
                                    index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

        else:
            self.PV = pd.Series(0, name=days[0] + '-' + days[-1],
                                index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days)*60*60*24))

            if self.has_pv:
                for k, day in enumerate(days):
                    start_index = str(k) + ' days'
                    end_index = start_index + ' 23:59:59'
                    self.PV[start_index:end_index] = self.pv_rated*pv_profile_generator(month)

                self.PV *= 1000

    def set_battery(self):
        p = self.P.sum(axis=1)
        if self.has_battery:

            battery_df = battery_function(p=p/1000, pv=self.PV/1000,
                                          soc_init=self.battery['soc_init'],
                                          soc_min=self.battery['soc_min'],
                                          soc_max=self.battery['soc_max'],
                                          e_max=self.battery['e_max'],
                                          ch_eff=self.battery['ch_eff'],
                                          dch_eff=self.battery['dch_eff'],
                                          p_max_bat=self.battery['p_max_bat'],
                                          t_lpf_bat=self.battery['t_lpf_bat']
                                          )

            self.battery['soc'] = battery_df['soc']
            self.battery['p_ch_bat'] = 1000*battery_df['p_ch_bat']
            self.battery['p_dch_bat'] = 1000*battery_df['p_dch_bat']
            self.total_grid_power = 1000*battery_df['grid_power']
        else:
            self.battery['soc'] = pd.Series(index=self.P.index, data=0)
            self.battery['p_ch_bat'] = pd.Series(index=self.P.index, data=0)
            self.battery['p_dch_bat'] = pd.Series(index=self.P.index, data=0)
            self.total_grid_power = self.PV - p

    def set_pv_and_battery(self, days, month):
        self.set_pv(days, month)
        self.set_battery()
        self.calculate_grid_power()

    def reset_pv_and_battery_values(self, df):

        self.battery = {
            key.lower(): None if np.isnan(df[key]) else df[key] for key in
                ['SoC_init', 'P_max_bat', 'E_max', 'SoC_min', 'SoC_max', 'ch_eff', 'dch_eff', 't_lpf_bat']
        }

        if self.battery['p_max_bat']:
            self.has_battery = True
        else:
            self.has_battery = False

        self.battery['soc'] = None
        self.battery['p_ch_bat'] = None
        self.battery['p_dch_bat'] = None

        self.pv_rated = None if np.isnan(df['PV_rated']) else df['PV_rated']

        if self.pv_rated:
            self.has_pv = True
        else:
            self.has_pv = False

        self.PV = None
        self.grid_power = pd.Series()
        self.total_grid_power = pd.Series()

    def calculate_grid_power(self):
        temp_pv = pd.DataFrame(index=self.P.index, columns=self.P.columns, data=0)
        temp_bat_ch = pd.DataFrame(index=self.P.index, columns=self.P.columns, data=0)
        temp_bat_dch = pd.DataFrame(index=self.P.index, columns=self.P.columns, data=0)
        if self.single_phase:
            temp_pv[self.selected_phase] = self.PV.values
            temp_bat_ch[self.selected_phase] = self.battery['p_ch_bat'].values
            temp_bat_dch[self.selected_phase] = self.battery['p_dch_bat'].values
        else:
            for phase in temp_pv.columns:
                temp_pv[phase] = self.PV.values/3
                temp_bat_ch[phase] = self.battery['p_ch_bat'].values/3
                temp_bat_dch[phase] = self.battery['p_dch_bat'].values/3

        self.grid_power = temp_pv - self.P - temp_bat_ch + temp_bat_dch

    def mean_auto_consumption_rate(self, battery=True):
        """
        Returns mean auto consumption rate. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: np.Series
        """
        if battery and self.has_battery:
            r = indexes.mean_auto_consumption_rate(gen=self.PV,  # + self.battery['p_dch_bat'],
                                                   injected=self.total_grid_power.clip(0))
        else:
            injected_power = self.PV - self.P.sum(axis=1)
            r = indexes.mean_auto_consumption_rate(gen=self.PV, injected=injected_power.clip(0))

        return r

    def cover_factors(self, battery=True):
        """
        Returns cover factors. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: (float, float)
        """
        if battery and self.has_battery:
            gamma_s, gamma_d = indexes.cover_factors(gen=self.PV + self.battery['p_dch_bat'],
                                                     load=self.P.sum(axis=1) + self.battery['p_ch_bat'])
        else:
            gamma_s, gamma_d = indexes.cover_factors(gen=self.PV, load=self.P.sum(axis=1))

        return gamma_s, gamma_d

    def load_match_index(self, interval='1h', battery=True):
        """
        Returns load match index. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        Parameter 'interval' is used for down-sampling.
        :param interval: str
        :param battery: bool
        :return: pd.Series
        """
        if battery and self.has_battery:
            lmi = indexes.load_match_index(gen=self.PV + self.battery['p_dch_bat'],
                                           load=self.P.sum(axis=1) + self.battery['p_ch_bat'],
                                           interval=interval)
        else:
            lmi = indexes.load_match_index(gen=self.PV, load=self.P.sum(axis=1), interval=interval)

        return lmi

    def loss_of_load_probability(self, battery=True):
        """
        Returns loss of load probability. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            lolp = indexes.loss_of_load_probability(gen=self.PV + self.battery['p_dch_bat'],
                                                    load=self.P.sum(axis=1) + self.battery['p_ch_bat'])
        else:
            lolp = indexes.loss_of_load_probability(gen=self.PV, load=self.P.sum(axis=1))

        return lolp

    def peaks_above_limit(self, p_limit=5000, battery=True):
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

    def no_grid_interaction_probability(self, period='15min', limit=0.001, battery=True):
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

    def one_percent_peak_power(self, period='1s', battery=True):
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

    def capacity_factor(self, battery=True):
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

    def self_consumption_rate(self, battery=True):
        """
        Returns self consumption rate. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            scr = indexes.self_consumption_rate(self.P.sum(axis=1), self.PV, battery_charge=self.battery['p_ch_bat'],
                                                battery_discharge=self.battery['p_dch_bat'])
        else:
            scr = indexes.self_consumption_rate(self.P.sum(axis=1), self.PV)

        return scr

    def self_sufficiency_rate(self, battery=True):
        """
        Returns self sufficiency rate. If 'battery' is False or the household has no battery,
        the index is calculated considering only PV.
        :param battery: bool
        :return: float
        """
        if battery and self.has_battery:
            ssr = indexes.self_sufficiency_rate(self.P.sum(axis=1), self.PV, battery_charge=self.battery['p_ch_bat'],
                                                battery_discharge=self.battery['p_dch_bat'])
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
            bui = indexes.battery_utilization_index(self.battery['p_ch_bat'], self.battery['p_dch_bat'],
                                                    self.battery['e_max'], depth_of_discharge=max(1-self.battery['soc']))
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
        p = self.P.sum(axis=1) + self.battery['p_ch_bat']
        q = self.Q.sum(axis=1)
        return p, q

    def get_total_production(self):
        """
        Returns power produced by PV and battery.
        :return: pd.Series
        """
        return self.PV + self.battery['p_dch_bat']

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

