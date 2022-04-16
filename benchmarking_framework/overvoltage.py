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

from sizing import pv_bess_sizing
from monte_carlo_sitting import random_sitting

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

MAXIMUM_SCENARIOS = overvoltage_dict['maximum_scenarios']

sizing_dict = overvoltage_dict['sizing_params']
sitting_dict = overvoltage_dict['sitting_params']


# DO NOT CHANGE THIS
# Parameters needed for battery and PV
ROWS = ['SoC_init', 'P_max_bat', 'E_max', 'SoC_min', 'SoC_max', 'PV_rated', 'ch_eff', 'dch_eff', 't_lpf_bat']


def main():

    # STEP 1
    # Create the SmartGrid object and set the simulation parameters as well as the grid power.
    grid = SmartGrid(csv_file=GRID_CSV, name='IEEE')
    grid.set_simulation_parameters(SIMULATION_FILE)
    grid.set_load_consumption()

    # read month in order to get pv power
    simulation_params = pd.read_csv(SIMULATION_FILE, index_col=0, header=None, squeeze=True).to_dict()
    month = simulation_params['month']
    pv_power = pv_profile_generator(month)

    # Read line limits as a dict
    line_limits = pd.read_csv(LINE_LIMITS_FILE, index_col=0, header=None, squeeze=True).to_dict()

    sizing_dict['normalized_pv'] = pv_power

    # Get house names
    houses = pd.read_csv(GRID_CSV, index_col=0).columns

    sitting_dict['houses'] = houses
    sitting_dict['pv_bes_parameters'] = ROWS

    # Initialize a dataframe for PV and battery parameters
    df = pd.DataFrame(columns=houses, index=ROWS)

    # for each home we should find pv_rated, battery maximum power and energy
    # initialize empty lists
    pv_rated_per_home = []
    p_bat_per_home = []
    e_bat_per_home = []

    # STEP 2
    for home in houses:
        sizing_dict['p'] = grid.homes[home].P.sum(axis=1) / 1000
        sizing_dict['single_phase'] = grid.homes[home].single_phase

        pv_rated, p_bat, e_bat = pv_bess_sizing(**sizing_dict)

        pv_rated_per_home.append(pv_rated)
        p_bat_per_home.append(p_bat)
        e_bat_per_home.append(e_bat)

    sitting_dict['pv_rated_per_home'] = pv_rated_per_home
    sitting_dict['e_bat_per_home'] = e_bat_per_home
    sitting_dict['p_bat_per_home'] = p_bat_per_home

    # STEP 3
    # In this loop we check random scenarios for over-voltage or thermal problems.

    attempts = 0
    while attempts < MAXIMUM_SCENARIOS:
        attempts += 1
        print('Attempt number {}'.format(attempts))

        # For each scenario, PV and batteries are randomly located.
        df, df_gen = random_sitting(**sitting_dict)

        # Apply the new PVs and batteries in order to get the new total power for each home
        grid.reset_pv_and_battery(df)
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

        # STEP 4
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


if __name__ == "__main__":
    main()

