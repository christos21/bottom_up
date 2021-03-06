import os.path
import pandas as pd
import opendssdirect as dss

from smart_grid import SmartGrid
from utils import sequential_voltage_vectors
from power_flow import calculate_passive_network_losses_from_power_flow


base_folder = 'final_scenarios/scenario_1/config_files'
droop_folder = 'final_scenarios/scenario_1/droop_config'
result_folder = 'final_scenarios/scenario_1/droop_power_flow_results'

# initialize the grid
grid_csv            = os.path.join(base_folder, 'grid_IEEE.csv')
simulation_file     = os.path.join(base_folder, 'simulation_parameters.csv')
generators_csv      = os.path.join(base_folder, 'generators.csv')
dss_path            = os.path.join(base_folder, 'IEEE.dss')
pv_battery_csv      = os.path.join(base_folder, 'pv_battery.csv')


prosumers_df = pd.read_csv(os.path.join(droop_folder, 'scenario_1_prosumers_df.csv'), index_col=0)
generators_df = pd.read_csv(os.path.join(droop_folder, 'scenario_1_generators_df.csv'), index_col=0)


# Create the original grid
grid = SmartGrid(csv_file=grid_csv, name='IEEE')
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()
df = pd.read_csv(pv_battery_csv, index_col=0)
df_gen = pd.read_csv(generators_csv, index_col=0)
grid.reset_pv_and_battery(df)
grid.set_generators(df_gen)



# Replace PV timeseries after curtailment
dss.run_command('Redirect ' + dss_path)
dss.run_command('solve')

load_names = list(grid.homes.keys())
generator_names = list(grid.generators.keys())

for i, load in enumerate(load_names):
    if grid.homes[load].has_pv:
        grid.homes[load].PV = prosumers_df.loc[:, load]
        grid.homes[load].set_battery()
        grid.homes[load].calculate_grid_power()

for i, generator in enumerate(generator_names):
    grid.generators[generator].P.iloc[:, 0] = generators_df.loc[:, generator].values/3
    grid.generators[generator].P.iloc[:, 1] = generators_df.loc[:, generator].values/3
    grid.generators[generator].P.iloc[:, 2] = generators_df.loc[:, generator].values/3


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



# passive power flow
calculate_passive_network_losses_from_power_flow(grid, dss_path=dss_path,
                                                 number_of_seconds=None, starting_second=0,
                                                 save_results=True, folder=result_folder, subsampling=10)
