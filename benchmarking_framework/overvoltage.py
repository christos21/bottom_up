"""
This script performs tests for PV and battery penetration on a smart grid and checks if there is overvoltage
or thermal problems at the lines.

Specifically, the user should fill all the necessary parameters in the configuration file overvoltage_config.yaml,
including the following files:

1) a csv file with appliances and appliance profiles for each home,
2) a csv file with the simulation parameters
3) a csv file with line limits
4) a dss file

NOTE: The name of each home in file 1 should be the bus where it is connected. This bus MUST appear in file 4.

The algorithm tries some random scenarios by installing PV and batteries to random households.
If there is an overvoltage and/or thermal problem, the results are saved as csv files.
"""

import os

import yaml
import numpy as np
import pandas as pd

from power_flow import solve_power_flow
from components.smart_grid import SmartGrid

from utils import pv_profile_generator, check_current_limits, sequential_voltage_vectors

with open('benchmarking_framework/overvoltage_config.yaml') as f:
    overvoltage_dict = yaml.safe_load(f)


# unpack parameters
GRID_CSV            = overvoltage_dict['input_files']['grid_csv']
SIMULATION_FILE     = overvoltage_dict['input_files']['simulation_file']
LINE_LIMITS_FILE    = overvoltage_dict['input_files']['line_limits_file']
DSS_FILE            = overvoltage_dict['input_files']['dss_file']


FOLDER_TO_SAVE_RESULTS = overvoltage_dict['output_files']['folder']
SOLAR_PARKS_FILE       = os.path.join(FOLDER_TO_SAVE_RESULTS, overvoltage_dict['output_files']['solar_parks'])
PV_BATTERY_FILE        = os.path.join(FOLDER_TO_SAVE_RESULTS, overvoltage_dict['output_files']['pv_battery'])
V1_FILE                = os.path.join(FOLDER_TO_SAVE_RESULTS, overvoltage_dict['output_files']['v1'])


POWER_FLOW_NUMBER_OF_SECONDS        = overvoltage_dict['power_flow_params']['number_of_seconds']
POWER_FLOW_STARTING_SECOND_OFFSET   = overvoltage_dict['power_flow_params']['starting_second_offset']


OVER_VOLTAGE_AND_NOT_THERMAL_LIMITS = overvoltage_dict['violations']['overvoltage_and_not_thermal']
OVER_VOLTAGE_AND_THERMAL_LIMITS     = overvoltage_dict['violations']['overvoltage_and_thermal']
OVER_VOLTAGE_PU_LIMIT               = overvoltage_dict['violations']['pu_limit']


SOC_INIT        = overvoltage_dict['battery_params']['soc_init']
SOC_MIN         = overvoltage_dict['battery_params']['soc_min']
SOC_MAX         = overvoltage_dict['battery_params']['soc_max']
CH_EFF          = overvoltage_dict['battery_params']['ch_eff']
DCH_EFF         = overvoltage_dict['battery_params']['dch_eff']
T_LPF_BAT       = overvoltage_dict['battery_params']['t_lpf']
BATTERY_MIN_KW  = overvoltage_dict['battery_params']['min_kW']
BATTERY_MAX_KW  = overvoltage_dict['battery_params']['max_kW']
BATTERY_MIN_KWH = overvoltage_dict['battery_params']['min_kWh']
BATTERY_MAX_KWH = overvoltage_dict['battery_params']['max_kW']


SINGLE_PHASE_PV_MAX_POWER               = overvoltage_dict['pv_params']['single_phase_pv_max_power']
THREE_PHASE_PV_MAX_POWER                = overvoltage_dict['pv_params']['three_phase_pv_max_power']
THREE_PHASE_PV_MIN_POWER                = overvoltage_dict['pv_params']['three_phase_pv_min_power']
PV_POWER_FOR_SOLAR_PARKS_AT_FREE_NODES  = overvoltage_dict['pv_params']['pv_power_for_solar_parks_at_free_nodes']


PROBABILITY_OF_PV                   = overvoltage_dict['probability_of_pv']
PROBABILITY_OF_BATTERY_GIVEN_PV     = overvoltage_dict['probability_of_battery_given_pv']
NUMBER_OF_SOLAR_PARKS_AT_FREE_NODES = overvoltage_dict['number_of_solar_parks_at_free_nodes']
AVAILABLE_NODES_FOR_SOLAR_PARKS     = overvoltage_dict['nodes_for_solar_parks']


MAXIMUM_SCENARIOS = overvoltage_dict['maximum_scenarios']


# DO NOT CHANGE THIS
# Parameters needed for battery and PV
ROWS = ['SoC_init', 'P_max_bat', 'E_max', 'SoC_min', 'SoC_max', 'PV_rated', 'ch_eff', 'dch_eff', 't_lpf_bat']


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
        if battery_p < BATTERY_MIN_KW:
            bat_p_max_per_home.append(0)
        else:
            bat_p_max_per_home.append(min(battery_p, BATTERY_MAX_KW))

        # Calculate the excess of energy produced by the PV, only during production hours
        energy_excess = np.sum((ratio * 1000 * pv_power - grid.homes[home].P.sum(axis=1).values) * (pv_power > 0)) / (
                    1000 * 3600)

        if energy_excess < BATTERY_MIN_KWH:
            bat_e_max_per_home.append(0)
        else:
            bat_e_max_per_home.append(min(np.ceil(energy_excess), BATTERY_MAX_KWH))

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
        gen_buses = np.random.choice(AVAILABLE_NODES_FOR_SOLAR_PARKS, NUMBER_OF_SOLAR_PARKS_AT_FREE_NODES, replace=False)
        df_gen = pd.DataFrame(columns=gen_buses, index=['phase_number', 'rated'])
        df_gen.loc['rated'] = PV_POWER_FOR_SOLAR_PARKS_AT_FREE_NODES
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
                          starting_second=t-POWER_FLOW_STARTING_SECOND_OFFSET,
                          folder=FOLDER_TO_SAVE_RESULTS)

    # Save csv files with information about PV, batteries and PV at free nodes
    df.to_csv(PV_BATTERY_FILE)
    df_gen.to_csv(SOLAR_PARKS_FILE)

    # save positive sequence voltage
    v1_df = pd.DataFrame(data=abs(V1), index=va.index)
    v1_df.to_csv(V1_FILE)

    # Save state of charge for all batteries
    socs = []
    cols = []
    for k, home in enumerate(grid.homes.keys()):
        if grid.homes[home].has_battery:
            cols.append(home)
            socs.append(grid.homes[home].battery['soc'].
                        iloc[t-POWER_FLOW_STARTING_SECOND_OFFSET:
                             t+POWER_FLOW_NUMBER_OF_SECONDS-POWER_FLOW_STARTING_SECOND_OFFSET].values)

    # socs_df = pd.DataFrame(columns=v1_df.index, index=cols, data=socs).T
    # socs_df.to_csv(os.path.join(FOLDER_TO_SAVE_RESULTS, 'soc.csv'))


if __name__ == "__main__":
    main()

