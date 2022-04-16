import os.path
import pandas as pd
import numpy as np
import random
import opendssdirect as dss

from utils import sequential_voltage_vectors
from components.smart_grid import SmartGrid


base_folder = 'final_scenarios/scenario_3/config_files'

result_folder = 'final_scenarios/scenario_3/power_flow_with_load_shifting_results'

# initialize the grid
grid_csv            = os.path.join(base_folder, 'grid_IEEE.csv')
simulation_file     = os.path.join(base_folder, 'simulation_parameters.csv')
generators_csv      = os.path.join(base_folder, 'generators.csv')
dss_path            = os.path.join(base_folder, 'IEEE.dss')
pv_battery_csv      = os.path.join(base_folder, 'pv_battery.csv')


# list of EVs that will be shifted
remove_evs = ['b7', 'b39', 'b56', 'b103', 'b170', 'b172', 'b204']

# Remove these EVs from the initial configuration df and keep the selected profile to ev_dict
ev_dict = {}
grid_df = pd.read_csv(grid_csv, index_col=0)
for node in grid_df.columns:
    if node in remove_evs:
        ev_dict[node] = grid_df.loc['BEV_profile', node]
        grid_df.loc['BEV', node] = 0


# Create the original grid without the selected EVs
grid = SmartGrid(grid_df=grid_df, name='IEEE')
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

# The selected EVs will be added to night hours after 02:00
list_start_times = []

# For each selected EV
for node in remove_evs:
    # load original profile
    profile = int(ev_dict[node])
    end_use = pd.read_csv(os.path.join('dataset/profiles/BEV/WD', 'profile_{}.csv'.format(profile)), index_col=0)

    # keep only the part of timeseries where EV is charging
    end_use_status = 1*(end_use > 0)
    start = np.where(end_use_status.diff() == 1)[0][0]
    end = np.where(end_use_status.diff() == -1)[0][0]
    new_end_use = end_use.iloc[start-1:end+1]

    # Choose a random starting hour
    potential_start_interval = grid[node].P.index[:2*60*60]
    start_time = random.choice(potential_start_interval)
    end_time = start_time + pd.Timedelta('{}s'.format(len(new_end_use)-1))
    list_start_times.append(start_time)

    # Add the EV charging power to the household power and to the grid total active power
    if grid[node].single_phase:
        phase = grid[node].selected_phase
        grid[node].P.loc[start_time:end_time, phase] += new_end_use.P.values
        grid[node].Q.loc[start_time:end_time, phase] += new_end_use.Q.values

        grid.P_load.loc[start_time:end_time, phase] += new_end_use.P.values
    else:
        for phase in ['a', 'b', 'c']:
            grid[node].P.loc[start_time:end_time, phase] += new_end_use.P.values / 3
            grid[node].Q.loc[start_time:end_time, phase] += new_end_use.Q.values / 3
            grid.P_load.loc[start_time:end_time, phase] += new_end_use.P.values / 3


# grid.P_load.sum(axis=1).to_csv('scenario_3_tr_load_after_shifting.csv')

# Add PV/BES units
df = pd.read_csv(pv_battery_csv, index_col=0)
df_gen = pd.read_csv(generators_csv, index_col=0)
grid.reset_pv_and_battery(df)
grid.set_generators(df_gen)


dss.run_command('Redirect ' + dss_path)
dss.run_command('solve')

load_names = list(grid.homes.keys())
generator_names = list(grid.generators.keys())

# solve power flow
grid.solve_power_flow(dss_path=dss_path,
                      folder=result_folder,
                      subsampling=10)


va = pd.read_csv(os.path.join(result_folder, 'Va_vec.csv'), index_col=0).astype(complex)
vb = pd.read_csv(os.path.join(result_folder, 'Vb_vec.csv'), index_col=0).astype(complex)
vc = pd.read_csv(os.path.join(result_folder, 'Vc_vec.csv'), index_col=0).astype(complex)

ia = pd.read_csv(os.path.join(result_folder, 'Ca.csv'), index_col=0)
ib = pd.read_csv(os.path.join(result_folder, 'Cb.csv'), index_col=0)
ic = pd.read_csv(os.path.join(result_folder, 'Cc.csv'), index_col=0)

V0, V1, V2 = sequential_voltage_vectors(va, vb, vc)

v0_df = pd.DataFrame(data=abs(V0), index=va.index)
v1_df = pd.DataFrame(data=abs(V1), index=va.index)
v2_df = pd.DataFrame(data=abs(V2), index=va.index)

# v0_df.to_csv(os.path.join(folder, 'V0.csv'))
v1_df.to_csv(os.path.join(result_folder, 'V1.csv'))
# v2_df.to_csv(os.path.join(folder, 'V2.csv'))

