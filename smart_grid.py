import pandas as pd
import warnings
import sys
import matplotlib.pyplot as plt
import json
import os

from smart_home import SmartHome
from utils import check_for_consecutive_days, pv_profile_generator, PHASES
from power_flow import solve_power_flow
from indexes import average_feeder_loading_index, substation_reserve_capacity, feeder_loss_to_load_ratio


class SmartGrid:

    def __init__(self, csv_file, name=None):
        df = pd.read_csv(csv_file, index_col=0)

        if name:
            self.name = name
        else:
            self.name = 'Smart grid 1'

        homes = df.columns
        self.int_to_home = {k: home for k, home in enumerate(homes)}

        self.homes = {}
        self.generators = {}

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
        if isinstance(home, str):
            return self.homes[home]
        else:
            return self.homes[self.int_to_home[home]]

    def set_simulation_parameters(self, simulation_file):
        sim_param = pd.read_csv(simulation_file, index_col=0, header=None, squeeze=True).to_dict()
        self.profile_option = sim_param['profile_option']
        self.month = sim_param['month']

        assert self.month in range(1, 13)

        consecutive, days = check_for_consecutive_days(sim_param)
        if not consecutive:
            warnings.warn('Warning. The input should be consecutive days.')
            sys.exit()
        self.days = days

    def set_load_consumption(self):

        assert self.month, 'No month is selected'
        assert len(self.days), 'Number of days is 0'

        for k, home in enumerate(self.homes.values()):
            if len(self.days) == 1:
                home.set_single_day_aggregated_profile(self.profile_option, self.days[0], self.month)
            else:
                home.set_multiple_days_aggregated_profile(self.days, self.month)

            home.set_pv_and_battery(self.days, self.month)

        self.P = self[0].grid_power.copy()
        self.P_load = self[0].P.copy()
        self.Q = self[0].Q.copy()
        for i in range(1, len(self.homes)):
            self.P += self[i].grid_power
            self.P_load += self[i].P
            self.Q += self[i].Q

    def reset_pv_and_battery(self, pv_and_battery_df):
        homes = pv_and_battery_df.columns

        for home in homes:
            if home not in self.homes.keys():
                print('No home named {}.'.format(home))
                continue

            self.homes[home].reset_pv_and_battery_values(pv_and_battery_df[home])
            self.homes[home].set_pv_and_battery(self.days, self.month)

        self.P = self[0].grid_power.copy()
        # self.P_load = self[0].P.copy()
        # self.Q = self[0].Q.copy()
        for i in range(1, len(self.homes)):
            self.P += self[i].grid_power
            # self.P_load += self[i].P
            # self.Q += self[i].Q

    def set_generators(self, csv_file):
        self.generators = {}

        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file, index_col=0)
        else:
            df = csv_file

        for pv in df.columns:
            self.generators[pv] = PV(pv, df[pv])
            self.generators[pv].set_power(self.days, self.month)

        for generator in self.generators.values():
            self.P += generator.P

    def solve_power_flow(self, dss_path, number_of_seconds=None, starting_second=0, save_results=True, folder=None):

        path_to_grid_params, path_to_results = solve_power_flow(self, dss_path=dss_path,
                                                                number_of_seconds=number_of_seconds,
                                                                starting_second=starting_second,
                                                                save_results=save_results,
                                                                folder=folder)
        self.path_to_grid_params = path_to_grid_params
        self.path_to_results = path_to_results

    def get_statistics(self):
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
        homes_with_pv = []
        for home_id, home_obj in self.homes.items():
            if home_obj.has_pv:
                homes_with_pv.append(home_id)

        return homes_with_pv

    def get_homes_with_battery(self):
        homes_with_battery = []
        for home_id, home_obj in self.homes.items():
            if home_obj.has_battery:
                homes_with_battery.append(home_id)

        return homes_with_battery

    def calculate_grid_indexes(self, path_to_grid_params=None, path_to_results=None):

        if path_to_results is None:
            path_to_results = self.path_to_results
        if path_to_grid_params is None:
            path_to_grid_params = self.path_to_grid_params

        if path_to_grid_params is None:
            print("Give path to grid parameters json file.")
            return
        if path_to_results is None:
            print("Give path to results folder.")
            return

        assert os.path.exists(path_to_grid_params)
        assert os.path.exists(path_to_results)

        with open(path_to_grid_params, 'r') as fp:
            params = json.load(fp)

        S_substation = params['S_substation']
        initial_bus = params['initial_bus']
        line_length = params['line_length']
        total_length = params['total_length']
        line_c = params['line_c']
        bus_names = params['bus_names']
        bus_connections = params['bus_connections']

        load_names = list(self.homes.keys())

        self.indexes['pbi'] = self.power_balance_index(bus_names, load_names, bus_connections, initial_bus)

        total_load = self.P_load.sum(axis=1)

        losses = pd.read_csv(os.path.join(path_to_results, 'Losses.csv'), index_col=0)
        total_load = total_load.iloc[:len(losses)]
        self.indexes['fllr'] = feeder_loss_to_load_ratio(total_load, losses.P)

        transformer_power = pd.read_csv(os.path.join(path_to_results, 'transformer_power.csv'), index_col=0)
        s = transformer_power['P_s'] + transformer_power['Q_s']
        self.indexes['src'] = substation_reserve_capacity(s, S_substation)

        line_power = pd.read_csv(os.path.join(path_to_results, 'line_power.csv'), index_col=0)
        self.indexes['afli'] = average_feeder_loading_index(line_power, line_length, line_c, total_length)

    def power_balance_index(self, bus_names, load_names, bus_connections, initial_bus):
        recursive_power_balance_index = {bus: None for bus in bus_names}

        for bus in bus_names:
            if bus in load_names:
                recursive_power_balance_index[bus] = self.homes[bus].total_grid_power

        def compute_pbi(bus, connections):
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
        p = self.P.sum(axis=1)
        q = self.Q.sum(axis=1)
        return p, q

    def plot_active_power(self, phase=None):
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

    def __init__(self, bus, pv_info):
        self.bus = bus
        self.pv_rated = pv_info['rated']
        self.P = None
        self.single_phase = True if pv_info['phase_number'] in [1, 2, 3] else False

        if self.single_phase:
            assert pv_info['phase_number'] in [1, 2, 3]
            self.selected_phase = PHASES[pv_info['phase_number']-1]
        else:
            self.selected_phase = None

    def set_power(self, days, month):
        self.P = pd.Series(0, name=days[0] + '-' + days[-1],
                           index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

        for k, day in enumerate(days):
            start_index = str(k) + ' days'
            end_index = start_index + ' 23:59:59'
            self.P[start_index:end_index] = self.pv_rated * pv_profile_generator(month)

        self.P *= 1000

        three_phase_p = pd.DataFrame(0, columns=['a', 'b', 'c'],
                                     index=pd.timedelta_range(start='00:00:00', freq='1s',
                                                              periods=len(days) * 60 * 60 * 24))

        if self.single_phase:
            three_phase_p[self.selected_phase] = self.P.values
        else:
            for phase in three_phase_p.columns:
                three_phase_p[phase] = self.P.values/3

        self.P = three_phase_p


