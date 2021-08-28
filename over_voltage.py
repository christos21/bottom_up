"""
This script performs tests for PV and battery penetration on a smart grid and checks if there is over-voltage
or thermal problems at the lines.

Specifically, the user should provide:
1) a csv file with appliances and appliance profiles for each home,
2) a csv file with the simulation parameters
3) a csv file with line limits
4) a dss file

NOTE: The name of each home in file 1 should be the bus where it is connected. This bus MUST appear in file 4.

Then the user should specify some parameters explained below and run the script.

The algorithm tries some random scenarios by installing PV and batteries to random households.
If there is an over-voltage and/or thermal problem, the results are saved as csv files.

"""

import os.path
from power_flow import solve_power_flow
from smart_grid import SmartGrid
from utils import pv_profile_generator, check_current_limits
import pandas as pd
import numpy as np


# specify grid, simulation and line limits csv paths as well as a dss file
GRID_CSV = 'test_cases/test_case_1_minor_over_voltage/grid_IEEE.csv'
SIMULATION_FILE = 'test_cases/test_case_1_minor_over_voltage/simulation_parameters.csv'
LINE_LIMITS_FILE = 'test_cases/line_limits.csv'
DSS_FILE = 'test_cases/test_case_1_minor_over_voltage/IEEE.dss'

# Folder where the results will be saved
FOLDER_TO_SAVE_RESULTS = 'over_voltage_07_08'

# Number of seconds to solve power flow
POWER_FLOW_NUMBER_OF_SECONDS = 60
POWER_FLOW_STARTING_SECOND_OFFSET = 30

# binary variable set to True if we want a scenario with over-voltage but no problem with line thermal limits
OVER_VOLTAGE_AND_NOT_THERMAL_LIMITS = True
# binary variable set to True if we want a scenario with both over-voltage and problem with line thermal limits
OVER_VOLTAGE_AND_THERMAL_LIMITS = True

# If none of the two binary variables is set to True, the algorithm will find a scenario with over-voltage regardless
# if there is a thermal problem or not

# pu limit for an over-voltage
OVER_VOLTAGE_PU_LIMIT = 1.10


# The following parameters will be fixed for all random scenarios
SOC_INIT = 0.1          # initial state of charge for all batteries
SOC_MIN = 0.1           # minimum state of charge
SOC_MAX = 0.9           # maximum state of charge
CH_EFF = 0.9            # charging efficiency
DCH_EFF = 0.9           # discharging efficiency
T_LPF_BAT = 100         # length of low pass filter


# minimum and maximum power (in kW) for single and three phase PV
SINGLE_PHASE_PV_MAX_POWER = 5
THREE_PHASE_PV_MAX_POWER = 20
THREE_PHASE_PV_MIN_POWER = 5

# PV power (kW) for PV installed at free nodes
PV_POWER_FOR_PV_AT_FREE_NODES = 20

battery_min_kW = 3      # battery minimum kW
battery_max_kW = 6      # battery maximum kW

battery_min_kWh = 3     # battery minimum kWh
battery_max_kWh = 20    # battery maximum kWh


# probability that a home has PV
PROBABILITY_OF_PV = 0.7
# probability that a home has a battery given that it has PV
PROBABILITY_OF_BATTERY_GIVEN_PV = 0.4

# Number of PVs that will be installed at free nodes
NUMBER_OF_PVS_AT_FREE_NODES = 5

# Maximum number of scenarios to check
MAXIMUM_SCENARIOS = 100

# free terminal nodes, where PV can be installed
term_nodes = ['b3', 'b14', 'b16', 'b25', 'b37', 'b38', 'b44', 'b68', 'b72', 'b74', 'b76', 'b78', 'b80', 'b82', 'b84',
              'b87', 'b88', 'b91', 'b93', 'b96', 'b100', 'b102', 'b106', 'b109', 'b111', 'b115', 'b117', 'b121', 'b123',
              'b126', 'b128', 'b129', 'b133', 'b139', 'b140', 'b142', 'b144', 'b145', 'b160', 'b176', 'b178', 'b180',
              'b182', 'b184', 'b187', 'b189', 'b191', 'b195', 'b197', 'b201', 'b202', 'b203']

# term_nodes = ['b202', 'b178', 'b144', 'b93', 'b37', 'b44', 'b160']


# DO NOT CHANGE THIS
# Parameters needed for battery and PV
ROWS = ['SoC_init', 'P_max_bat', 'E_max', 'SoC_min', 'SoC_max', 'PV_rated', 'ch_eff', 'dch_eff', 't_lpf_bat']


def sequential_voltage_vectors(va, vb, vc):
    """
    Calculates positive, zero and negative sequence voltages.

    Voltages are considered as dataframes with each column corresponding to a bus and each row to a time instant.

    :param va: pd.DataFrame
    :param vb: pd.DataFrame
    :param vc: pd.DataFrame
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame
    """
    va.iloc[:, 0] = va.iloc[:, 0] / (11 / 0.416)
    va = va / (416 / np.sqrt(3))

    vb.iloc[:, 0] = vb.iloc[:, 0] / (11 / 0.416)
    vb = vb / (416 / np.sqrt(3))

    vc.iloc[:, 0] = vc.iloc[:, 0] / (11 / 0.416)
    vc = vc / (416 / np.sqrt(3))

    # calculate sequential voltages vector
    V0 = (va + vb + vc) / 3
    V1 = (va + vb * np.exp(1j * 2 * np.pi / 3) + vc * np.exp(1j * 4 * np.pi / 3)) / 3
    V2 = (va + vb * np.exp(1j * 4 * np.pi / 3) + vc * np.exp(1j * 2 * np.pi / 3)) / 3

    return V0, V1, V2


def main():

    # Create the SmartGrid object and set the simulation parameters as well as the grid power.
    grid = SmartGrid(csv_file=GRID_CSV, name='IEEE')
    grid.set_simulation_parameters(SIMULATION_FILE)
    grid.set_load_consumption()

    # Read line limits as a dict
    line_limits = pd.read_csv(LINE_LIMITS_FILE, index_col=0, header=None, squeeze=True).to_dict()

    # read month in order to get pv power
    simulation_params = pd.read_csv(SIMULATION_FILE, index_col=0, header=None, squeeze=True).to_dict()
    month = simulation_params['month']

    # read pv profile to get mean daily production
    pv_power = pv_profile_generator(month)
    mean_daily_pv_power = np.mean(pv_power)

    # Get house names
    df = pd.read_csv(GRID_CSV, index_col=0)
    cols = df.columns

    # Initialize a dataframe for PV and battery parameters
    df = pd.DataFrame(columns=cols, index=ROWS)

    # for each home we should find pv_rated, battery maximum power and energy

    # initialize empty lists
    pv_rated_per_home = []
    bat_p_max_per_home = []
    bat_e_max_per_home = []

    for home in cols:
        # Calculate pv_rated for each home as the mean daily consumption over mean daily production
        home_mean_daily_power = grid.homes[home].P.sum(axis=1).mean() / 1000
        ratio = home_mean_daily_power / mean_daily_pv_power
        ratio = np.ceil(ratio)

        # apply corresponding limits for single and three-phase PV
        if grid.homes[home].single_phase:
            pv_rated = min(ratio, SINGLE_PHASE_PV_MAX_POWER)
        else:
            pv_rated = min(THREE_PHASE_PV_MAX_POWER, max(ratio, THREE_PHASE_PV_MIN_POWER))

        pv_rated_per_home.append(pv_rated)

        # Battery power is half the PV power
        battery_p = np.ceil(pv_rated / 2)
        if battery_p < battery_min_kW:
            bat_p_max_per_home.append(0)
        else:
            bat_p_max_per_home.append(min(battery_p, battery_max_kW))

        # Calculate the excess of energy produced by the PV, only during production hours
        energy_excess = np.sum((ratio * 1000 * pv_power - grid.homes[home].P.sum(axis=1).values) * (pv_power > 0)) / (
                    1000 * 3600)

        if energy_excess < battery_min_kWh:
            bat_e_max_per_home.append(0)
        else:
            bat_e_max_per_home.append(min(np.ceil(energy_excess), battery_max_kWh))

    for i in range(len(bat_p_max_per_home)):
        # in case the maximum battery charging power is higher than half the capacity of the battery, set the
        # battery charging power to capacity/2.
        if bat_p_max_per_home[i] > bat_e_max_per_home[i] / 2:
            bat_p_max_per_home[i] = bat_e_max_per_home[i] / 2

    # In this loop we check random scenarios for over-voltage or thermal problems.
    # For each scenario, PV and batteries are randomly located.

    attempts = 0
    while attempts < MAXIMUM_SCENARIOS:
        attempts += 1
        print('Attempt number {}'.format(attempts))

        # re-initialize the df inside the loop
        df = pd.DataFrame(columns=cols, index=ROWS)

        # for each home
        for k, home in enumerate(cols):
            # Set randomly PV based on the probability but pv_rated should be > 0
            if np.random.random() < PROBABILITY_OF_PV and pv_rated_per_home[k] > 0:

                df.loc['PV_rated', home] = pv_rated_per_home[k]

                # Set randomly battery based on the probability but the home MUST have PV
                if np.random.random() < PROBABILITY_OF_BATTERY_GIVEN_PV and bat_e_max_per_home[k] > 0 \
                        and bat_p_max_per_home[k] > 0:

                    df.loc['SoC_init', home] = SOC_INIT
                    df.loc['SoC_min', home] = SOC_MIN
                    df.loc['SoC_max', home] = SOC_MAX
                    df.loc['ch_eff', home] = CH_EFF
                    df.loc['dch_eff', home] = DCH_EFF
                    df.loc['t_lpf_bat', home] = T_LPF_BAT

                    df.loc['P_max_bat', home] = bat_p_max_per_home[k]
                    df.loc['E_max', home] = bat_e_max_per_home[k]

        # Apply the new PVs and batteries in order to get the new total power for each home
        grid.reset_pv_and_battery(df)

        # Add random generators
        gen_buses = np.random.choice(term_nodes, NUMBER_OF_PVS_AT_FREE_NODES, replace=False)
        df_gen = pd.DataFrame(columns=gen_buses, index=['phase_number', 'rated'])
        df_gen.loc['rated'] = PV_POWER_FOR_PV_AT_FREE_NODES
        grid.set_generators(df_gen)

        # for over-voltage check when grid power (production-load) is maximum
        t = np.argmax(grid.P.sum(axis=1))

        # solve power flow
        va, vb, vc, ia, ib, ic, _ = solve_power_flow(grid, dss_path=DSS_FILE,
                                                     number_of_seconds=POWER_FLOW_NUMBER_OF_SECONDS,
                                                     starting_second=t-POWER_FLOW_STARTING_SECOND_OFFSET,
                                                     save_results=False)

        # find positive sequence voltage
        _, V1, _ = sequential_voltage_vectors(va, vb, vc)
        V1_magn = abs(V1)

        print('Maximum V1 {}'.format(np.max(V1_magn.values)))

        # Check thermal limits for all phases
        thermal_limits = check_current_limits(ia, line_limits) and check_current_limits(ib, line_limits) and \
                         check_current_limits(ic, line_limits)

        # Check the necessary conditions
        if OVER_VOLTAGE_AND_THERMAL_LIMITS:
            pass_flag = (V1_magn > OVER_VOLTAGE_PU_LIMIT).sum().sum() and not thermal_limits
        elif OVER_VOLTAGE_AND_NOT_THERMAL_LIMITS:
            pass_flag = (V1_magn > OVER_VOLTAGE_PU_LIMIT).sum().sum() and thermal_limits
        else:
            pass_flag = (V1_magn > OVER_VOLTAGE_PU_LIMIT).sum().sum()

        # if the conditions are met, break the while loop
        if pass_flag:
            break

    # solve power flow again but now save results
    grid.solve_power_flow(dss_path=DSS_FILE, number_of_seconds=POWER_FLOW_NUMBER_OF_SECONDS,
                          starting_second=t-POWER_FLOW_STARTING_SECOND_OFFSET, save_results=True,
                          folder=FOLDER_TO_SAVE_RESULTS)

    # Save csv files with information about PV, batteries and PV at free nodes
    df.to_csv(os.path.join(FOLDER_TO_SAVE_RESULTS, 'pv_battery.csv'))
    df_gen.to_csv(os.path.join(FOLDER_TO_SAVE_RESULTS, 'generators.csv'))

    # save positive sequence voltage
    v1_df = pd.DataFrame(data=abs(V1), index=va.index)
    v1_df.to_csv(os.path.join(FOLDER_TO_SAVE_RESULTS, 'V1.csv'))

    # Save state of charge for all batteries
    socs = []
    cols = []
    for k, home in enumerate(grid.homes.keys()):
        if grid.homes[home].has_battery:
            cols.append(home)
            socs.append(grid.homes[home].battery['soc'].
                        iloc[t-POWER_FLOW_STARTING_SECOND_OFFSET:
                             t+POWER_FLOW_NUMBER_OF_SECONDS-POWER_FLOW_STARTING_SECOND_OFFSET].values)

    socs_df = pd.DataFrame(columns=v1_df.index, index=cols, data=socs).T
    socs_df.to_csv(os.path.join(FOLDER_TO_SAVE_RESULTS, 'soc.csv'))


if __name__ == "__main__":
    main()

