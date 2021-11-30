import pandas as pd
import warnings
import sys
import matplotlib.pyplot as plt
import json
import os
from scipy.io import loadmat

from smart_home import SmartHome
from utils import check_for_consecutive_days, pv_profile_generator, PHASES
from power_flow import solve_power_flow, calculate_passive_network_losses_from_power_flow
from indexes import substation_reserve_capacity, feeder_loss_to_load_ratio, voltage_unbalance_factors, \
    losses_to_load_ratio, losses_reduction_index, mean_voltage_variance, voltage_level_quantification_index, \
    average_feeding_loading_index


class SmartGrid:
    """
    Class representing a grid of smart homes and solar panels.
    """

    def __init__(self, grid_df=None, csv_file=None, name=None):
        """
        Initialization of the smart grid based on a csv file as explained in the manual.
        We can either pass the path to the csv file at parameter csv_file OR
        the pandas dataframe at parameter grid_df.
        :param grid_df: pd.DataFrame
        :param csv_file: str
        :param name: str
        """
        # Get the df either from csv_file or from grid_df
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file, index_col=0)
        elif isinstance(grid_df, pd.DataFrame):
            df = grid_df
        else:
            warnings.warn('Wrong input for SmartGrid initialization.')
            sys.exit()

        if name:
            self.name = name
        else:
            self.name = 'Smart grid 1'

        homes = df.columns
        # map homes with integer numbers
        self.int_to_home = {k: home for k, home in enumerate(homes)}

        self.homes = {}
        self.generators = {}

        # Represent each home as a SmartHome object.
        for home in homes:
            self.homes[home] = SmartHome(home, df[home])

        self.profile_option = None
        self.month = None
        self.days = []

        self.P = None
        self.P_load = None
        self.Q = None
        self.PV = pd.Series()

        self.indexes = {}

        self.path_to_grid_params = None
        self.path_to_results = None

    def __getitem__(self, home):
        """
        For parsing the grid as a list or dictionary.
        :param home: str | int
        :return:
        """
        if isinstance(home, str):
            return self.homes[home]
        elif isinstance(home, int):
            return self.homes[self.int_to_home[home]]

    def set_simulation_parameters(self, simulation_file):
        """
        Reads the simulation parameters from a csv file.
        :param simulation_file: str
        :return:
        """
        sim_param = pd.read_csv(simulation_file, index_col=0, header=None, squeeze=True).to_dict()
        self.profile_option = sim_param['profile_option']
        self.month = sim_param['month']

        assert self.month in range(1, 13)

        # Check that the days in the simulation_file are consecutive.
        # For example, we can not simulate Monday and Wednesday without Tuesday.
        consecutive, days = check_for_consecutive_days(sim_param)
        if not consecutive:
            warnings.warn('Warning. The input should be consecutive days.')
            sys.exit()
        self.days = days

    def set_load_consumption(self):
        """
        This method creates the load profile for each home.
        :return:
        """

        assert self.month, 'No month is selected'
        assert len(self.days), 'Number of days is 0'

        # Iterate through all homes
        for k, home in enumerate(self.homes.values()):
            # Call the correct method based on the number of days.
            if len(self.days) == 1:
                home.set_single_day_aggregated_profile(self.profile_option, self.days[0])
            else:
                home.set_multiple_days_aggregated_profile(self.days)

            # Set PV and battery power.
            home.set_pv_and_battery(self.days, self.month)

        # Sum aggregated P, P_load and Q (per phase) for all homes.
        self.P = self[0].grid_power.copy()
        self.P_load = self[0].P.copy()
        self.Q = self[0].Q.copy()
        for i in range(1, len(self.homes)):
            self.P += self[i].grid_power
            self.P_load += self[i].P
            self.Q += self[i].Q

    def set_load_consumption_from_mat_file(self, mat_file):
        """
        This method creates the load profile for each home which are saved in a .mat file.
        :return:
        """
        data = loadmat(mat_file)

        p = data['Pload']
        q = data['Qload']

        phases = data['phase_con']

        assert len(self.days) == p.shape[0]//(60*60*24), 'Number of days in mat file and simulation parameters ' \
                                                         'are different'

        pv = data['pv']

        homes = data['PQLoadname']

        column = 0

        for k, home in enumerate(homes):
            home = home.strip()
            assert self.homes[home]
            if self.homes[home].single_phase:
                phase = self.homes[home].selected_phase
                assert phase == phases[k].strip(), 'Different phases in .mat and grid csv file for home {}'.format(home)
                home_p = p[:, column]
                home_q = q[:, column]

                column += 1

                self.homes[home].set_load_from_arrays(p_array=home_p, q_array=home_q, phase=phase, days=self.days)

            else:

                assert phases[k].strip() == 'abc', 'Different phases in .mat and grid csv file for home {}'.format(home)
                home_p = p[:, column:column+3]
                home_q = q[:, column:column+3]

                column += 3

                self.homes[home].set_load_from_arrays(p_array=home_p, q_array=home_q, days=self.days)

            self.homes[home].set_pv_and_battery(days=self.days, month=self.month, pv_from_array=True, pv_array=pv)

        # Sum aggregated P, P_load and Q (per phase) for all homes.
        self.P = self[0].grid_power.copy()
        self.P_load = self[0].P.copy()
        self.Q = self[0].Q.copy()
        for i in range(1, len(self.homes)):
            self.P += self[i].grid_power
            self.P_load += self[i].P
            self.Q += self[i].Q

    def reset_pv_and_battery(self, pv_and_battery_df, pv_from_array=False, pv_array=None):
        """
        Method that resets PV and battery properties for all homes.
        The time-series for PV production and battery charging/discharging are also reset.
        A new dataframe is needed including the properties of the new PV and batteries.
        :param pv_and_battery_df: pd.DataFrame
        :param pv_from_array: bool
        :param pv_array: np.array
        :return:
        """

        homes = pv_and_battery_df.columns

        for home in homes:
            if home not in self.homes.keys():
                print('No home named {}.'.format(home))
                continue

            self.homes[home].reset_pv_and_battery_values(pv_and_battery_df[home])
            self.homes[home].set_pv_and_battery(self.days, self.month, pv_from_array, pv_array)

        # Sum aggregated P (per phase) for all homes. P_load and Q remain the same since the new PV and
        # batteries do not affect the load or the reactive power.
        self.P = self[0].grid_power.copy()

        for i in range(1, len(self.homes)):
            self.P += self[i].grid_power

    def set_generators(self, pv_df=None, csv_file=None, pv_from_array=False, pv_array=None):
        """
        Method for setting solar panels at free nodes of the grid.
        We can either pass the path to the csv file at parameter csv_file OR
        the pandas dataframe at parameter pv_df.
        :param pv_df: pd.DataFrame | None
        :param csv_file: str | None
        :param pv_from_array: bool
        :param pv_array: np.array
        :return:
        """
        self.generators = {}

        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file, index_col=0)
        elif isinstance(pv_df, pd.DataFrame):
            df = pv_df
        else:
            warnings.warn('Wrong input.')
            sys.exit()

        for pv in df.columns:
            self.generators[pv] = PV(pv, df[pv])
            self.generators[pv].set_power(self.days, self.month, from_array=pv_from_array, pv_array=pv_array)

        for generator in self.generators.values():
            self.P += generator.P

    def solve_power_flow(self, dss_path, folder, number_of_seconds=None, starting_second=0, subsampling=1, home_power_factor=None):
        """
        Method for solving power flow for the grid.
        :param dss_path: str, Path to a .dss file. The name of each home should be the name
                              of the bus that is connected to.
        :param number_of_seconds: int, Number of seconds to solve the power flow
        :param starting_second: int, The first second for which the power flow will be solved.
        :param folder: str, Path of the folder where the results will be saved.
        :param subsampling: int
        :return:
        """

        solve_power_flow(self, dss_path=dss_path,
                         number_of_seconds=number_of_seconds,
                         starting_second=starting_second,
                         save_results=True,
                         folder=folder,
                         subsampling=subsampling,
                         home_power_factor=home_power_factor)

        self.path_to_results = folder

    def solve_passive_power_flow(self, dss_path, folder, number_of_seconds=None, starting_second=0,
                                 subsampling=1, home_power_factor=None):
        """
        Method for solving power flow for the grid assuming no PV and batteries.
        :param dss_path: str, Path to a .dss file. The name of each home should be the name
                              of the bus that is connected to.
        :param number_of_seconds: int, Number of seconds to solve the power flow
        :param starting_second: int, The first second for which the power flow will be solved.
        :param folder: str, Path of the folder where the results will be saved if save_results is True.
        :param subsampling: int
        :return:
        """
        calculate_passive_network_losses_from_power_flow(self, dss_path=dss_path,
                                                         number_of_seconds=number_of_seconds,
                                                         starting_second=starting_second,
                                                         save_results=True,
                                                         folder=folder,
                                                         subsampling=subsampling,
                                                         home_power_factor=home_power_factor)

    def get_statistics(self):
        """
        Method that returns
            - the number of homes with battery and PV
            - number of three-phase homes and single-phase homes connected to each phase
            - number of homes that includes each appliance
        :return: dict, dict
        """
        pv = 0
        battery = 0
        appliances = {}
        three_phase = 0
        single_phase = {PHASES[0]: 0, PHASES[1]: 0, PHASES[2]: 0}

        for home in self.homes.values():
            if home.single_phase:
                single_phase[home.selected_phase] += 1
            else:
                three_phase += 1

            if home.has_pv:
                pv += 1

            if home.has_battery:
                battery += 1

            for app in home.simulation_appliances:
                if app not in appliances.keys():
                    appliances[app] = 1
                else:
                    appliances[app] += 1

        stats_dict = {'pv': pv, 'battery': battery, 'three_phase': three_phase,
                      'phase_a': single_phase[PHASES[0]], 'phase_b': single_phase[PHASES[1]],
                      'phase_c': single_phase[PHASES[2]]}

        return stats_dict, appliances

    def get_homes_with_pv(self):
        """
        Returns list of households with a PV.
        :return: [str]
        """
        homes_with_pv = []
        for home_id, home_obj in self.homes.items():
            if home_obj.has_pv:
                homes_with_pv.append(home_id)

        return homes_with_pv

    def get_homes_with_battery(self):
        """
        Returns list of households with a battery energy storage system.
        :return: [str]
        """
        homes_with_battery = []
        for home_id, home_obj in self.homes.items():
            if home_obj.has_battery:
                homes_with_battery.append(home_id)

        return homes_with_battery

    def calculate_grid_indexes(self, path_to_grid_params, path_to_results, line_limits_file):
        """
        Calculates some grid-related indexes for PV-BESS assessment.
        For the calculation of these indexes, power flow should be solved and the results should be stored.
        The input parameters are the path of the result folder and the path of the json file
        created by the solve_power_flow method.

        :param path_to_grid_params: str
        :param path_to_results: str
        :return:
        """

        assert os.path.exists(path_to_grid_params)
        assert os.path.exists(path_to_results)

        with open(path_to_grid_params, 'r') as fp:
            params = json.load(fp)

        # get grid parameters
        S_substation = params['S_substation']
        initial_bus = params['initial_bus']
        bus_names = params['bus_names']
        bus_connections = params['bus_connections']
        r_to_bus = params['resistance_from_start_to_each_bus']
        length_to_bus = params['length_from_start_to_each_bus']
        line_length = params['line_length']

        load_names = list(self.homes.keys())

        losses = pd.read_csv(os.path.join(path_to_results, 'Losses.csv'), index_col=0)
        line_power = pd.read_csv(os.path.join(path_to_results, 'line_power.csv'), index_col=0)
        Va = pd.read_csv(os.path.join(path_to_results, 'Va.csv'), index_col=0)
        Vb = pd.read_csv(os.path.join(path_to_results, 'Vb.csv'), index_col=0)
        Vc = pd.read_csv(os.path.join(path_to_results, 'Vc.csv'), index_col=0)

        Va_vec = pd.read_csv(os.path.join(path_to_results, 'Va_vec.csv'), index_col=0).astype(complex)
        Vb_vec = pd.read_csv(os.path.join(path_to_results, 'Vb_vec.csv'), index_col=0).astype(complex)
        Vc_vec = pd.read_csv(os.path.join(path_to_results, 'Vc_vec.csv'), index_col=0).astype(complex)

        transformer_power = pd.read_csv(os.path.join(path_to_results, 'transformer_power.csv'), index_col=0)

        passive_losses = pd.read_csv(os.path.join(path_to_results, 'Losses_for_passive_network.csv'), index_col=0)
        passive_va = pd.read_csv(os.path.join(path_to_results, 'Va_for_passive_network.csv'), index_col=0)
        passive_vb = pd.read_csv(os.path.join(path_to_results, 'Vb_for_passive_network.csv'), index_col=0)
        passive_vc = pd.read_csv(os.path.join(path_to_results, 'Vc_for_passive_network.csv'), index_col=0)

        passive_va_vec = pd.read_csv(os.path.join(path_to_results, 'Va_vec_for_passive_network.csv'), index_col=0).astype(complex)
        passive_vb_vec = pd.read_csv(os.path.join(path_to_results, 'Vb_vec_for_passive_network.csv'), index_col=0).astype(complex)
        passive_vc_vec = pd.read_csv(os.path.join(path_to_results, 'Vc_vec_for_passive_network.csv'), index_col=0).astype(complex)

        # calculate power balance index
        self.indexes['pbi'] = self.power_balance_index(bus_names, load_names, bus_connections, initial_bus)

        # calculate feeder loss to load ratio
        total_load = self.P_load.sum(axis=1)
        total_load = total_load.loc[list(losses.index)]  #total_load.iloc[:len(losses)]
        self.indexes['fllr'] = feeder_loss_to_load_ratio(total_load, losses.P)

        # calculate substation reserve capacity
        s = (transformer_power['P_s']**2 + transformer_power['Q_s']**2)**(1/2)
        self.indexes['src'] = substation_reserve_capacity(s, S_substation)

        # calculate VUF0, VUF2
        self.indexes['VUF0'], self.indexes['VUF2'] = voltage_unbalance_factors(Va_vec, Vb_vec, Vc_vec)

        # calculate lri and llr
        self.indexes['llr'] = losses_to_load_ratio(total_load, losses.P)
        self.indexes['lri'] = losses_reduction_index(losses.P, passive_losses.P)

        # sigma
        self.indexes['mean_voltage_variance_va'] = mean_voltage_variance(Va)
        self.indexes['mean_voltage_variance_vb'] = mean_voltage_variance(Vb)
        self.indexes['mean_voltage_variance_vc'] = mean_voltage_variance(Vc)


        # vlqi
        self.indexes['vlqi'] = voltage_level_quantification_index(Va_vec, Vb_vec, Vc_vec, passive_va_vec, passive_vb_vec, passive_vc_vec, r_to_bus)

        # afli
        line_limits = pd.read_csv(line_limits_file, index_col=0, header=None, squeeze=True).to_dict()
        self.indexes['afli'] = average_feeding_loading_index(length_to_bus, line_length, line_limits, line_power)


    def power_balance_index(self, bus_names, load_names, bus_connections, initial_bus):
        """
        Calculates power balance index as described in
        'M. Hasheminamin, V. G. Agelidis, V. Salehi, R. Teodorescu and B. Hredzak, "Index-Based Assessment of Voltage
        Rise and Reverse Power Flow Phenomena in a Distribution Feeder Under High PV Penetration," in IEEE Journal of
        Photovoltaics, vol. 5, no. 4, pp. 1158-1168, July 2015'

        :param bus_names: [str]
        :param load_names: [str]
        :param bus_connections: dict {str: [str]}
        :param initial_bus: str
        :return: dict {str: float}
        """
        recursive_power_balance_index = {bus: None for bus in bus_names}

        for bus in bus_names:
            if bus in load_names:
                recursive_power_balance_index[bus] = self.homes[bus].total_grid_power

        def compute_pbi(bus, connections):
            """
            Recursive function for pbi computation.
            :param bus: str
            :param connections: [str]
            :return: float
            """
            pbi = 0

            for connection in connections:
                if recursive_power_balance_index[connection] is None:
                    needed_buses = set(bus_connections[connection]) - {bus}
                    temp = compute_pbi(connection, list(needed_buses))

                    if connection in list(self.generators.keys()):
                        temp += self.generators[connection].P.sum(axis=1)

                    recursive_power_balance_index[connection] = temp

                pbi += recursive_power_balance_index[connection]

            return pbi

        recursive_power_balance_index[initial_bus] = compute_pbi(initial_bus, bus_connections[initial_bus])

        return recursive_power_balance_index

    def get_power(self):
        """
        Returns total active power and reactive power.
        :return: pd.Series, pd.Series
        """
        p = self.P.sum(axis=1)
        q = self.Q.sum(axis=1)
        return p, q

    def plot_active_power(self, phase=None):
        """
        Plots the active power.
        If 'phase' is None, the total active power is plotted.
        If 'phase' is 'a', 'b' or 'c' the corresponding phase is plotted.
        :param phase: None | 'a' | 'b' | 'c'
        :return:
        """
        plt.figure()
        if phase:
            self.P[phase].plot()
            plt.title(self.name + ' phase ' + phase + ' - Active power', fontsize=20)
        else:
            p, _ = self.get_power()
            p.plot()
            plt.title(self.name + ' - Total active power', fontsize=20)
        plt.show()

    def plot_reactive_power(self, phase=None):
        """
        Plots the reactive power.
        If 'phase' is None, the total reactive power is plotted.
        If 'phase' is 'a', 'b' or 'c' the corresponding phase is plotted.
        :param phase: None | 'a' | 'b' | 'c'
        :return:
        """
        plt.figure()
        if phase:
            self.Q[phase].plot()
            plt.title(self.name + ' phase ' + phase + ' - Reactive power', fontsize=20)
        else:
            _, q = self.get_power()
            q.plot()
            plt.title(self.name + ' - Total reactive power', fontsize=20)
        plt.show()

    def plot_power(self, phase=None):
        """
        Plots both the active and the reactive power.
        If 'phase' is None, the total power is plotted.
        If 'phase' is 'a', 'b' or 'c' the corresponding phase is plotted.
        :param phase: None | 'a' | 'b' | 'c'
        :return:
        """
        plt.figure()
        if phase:
            self.P[phase].plot()
            self.Q[phase].plot()
            plt.title(self.name + ' phase ' + phase, fontsize=20)
            plt.legend(['Active power', 'Reactive power'], fontsize=18)
        else:
            p, q = self.get_power()
            p.plot()
            q.plot()
            plt.title(self.name, fontsize=20)
            plt.legend(['Active power', 'Reactive power'], fontsize=18)
        plt.show()


class PV:
    """
    Class representing solar panel at free node of the grid.
    """

    def __init__(self, bus, pv_info):
        """
        Initialization of the PV based on a dataframe as explained in the manual.
        These type of PV are always three-phase but in this project we can create single-phase as well.

        :param bus: str, Name of the bus that the panel is connected to.
        :param pv_info: pd.DataFrame
        """
        self.bus = bus
        self.pv_rated = pv_info['rated']
        self.P = None
        self.single_phase = True if pv_info['phase_number'] in [1, 2, 3] else False

        if self.single_phase:
            assert pv_info['phase_number'] in [1, 2, 3]
            self.selected_phase = PHASES[pv_info['phase_number']-1]
        else:
            self.selected_phase = None

    def set_power(self, days, month, from_array=False, pv_array=None):
        """
        This method creates the solar production based on the month.
        :param days: [str]
        :param month: int
        :param from_array: bool
        :param pv_array: np.array
        :return:
        """
        if from_array:
            pv_values = pv_array
        else:
            pv_values = pv_profile_generator(month)

        # initialize the total production series with 0s
        self.P = pd.Series(0, name=days[0] + '-' + days[-1],
                           index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

        # Calculate production for each day
        for k, day in enumerate(days):
            start_index = str(k) + ' days'
            end_index = start_index + ' 23:59:59'
            self.P[start_index:end_index] = self.pv_rated * pv_values

        self.P *= 1000

        # initialize the production dataframe (per phase) with 0s
        three_phase_p = pd.DataFrame(0, columns=['a', 'b', 'c'],
                                     index=pd.timedelta_range(start='00:00:00', freq='1s',
                                                              periods=len(days) * 60 * 60 * 24))

        # if the panel is single-phase assign the total production to the correct phase
        if self.single_phase:
            three_phase_p[self.selected_phase] = self.P.values

        # if the panel is three-phase split between the phases
        else:
            for phase in three_phase_p.columns:
                three_phase_p[phase] = self.P.values/3

        self.P = three_phase_p


