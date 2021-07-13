import os.path
from power_flow import solve_power_flow
from smart_grid import SmartGrid
from utils import pv_profile_generator
import pandas as pd
import numpy as np

# initialize the grid
grid_csv = 'test_cases/grid_IEEE.csv'
simulation_file = 'test_grids/simulation_parameters.csv'

result_folder = 'over-voltage-scenario'

grid = SmartGrid(grid_csv, name='IEEE')
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

df = pd.read_csv(os.path.join(result_folder, 'pv_battery.csv'), index_col=0)
df_gen = pd.read_csv(os.path.join(result_folder, 'generators.csv'), index_col=0)

grid.reset_pv_and_battery(df)
grid.set_generators(df_gen)



# for over-voltage check when argmax(grid.P) Q: sum for all phases??
t = np.argmax(grid.P.sum(axis=1))
# then
# grid.solve_power_flow(dss_path, number_of_seconds=60, starting_second=argmax, save_results=False, folder=None)

va, vb, vc = solve_power_flow(grid, dss_path='test_grids/IEEE.dss', number_of_seconds=60, starting_second=t-30,
                              save_results=False)

va.iloc[:, 0] = va.iloc[:, 0]/(11/0.416)
va = va/(416/np.sqrt(3))

vb.iloc[:, 0] = vb.iloc[:, 0]/(11/0.416)
vb = vb/(416/np.sqrt(3))

vc.iloc[:, 0] = vc.iloc[:, 0]/(11/0.416)
vc = vc/(416/np.sqrt(3))

# calculate sequential voltages vector
V0 = (va + vb + vc) / 3
V1 = (va + vb * np.exp(1j * 2 * np.pi / 3) + vc * np.exp(1j * 4 * np.pi / 3)) / 3
V2 = (va + vb * np.exp(1j * 4 * np.pi / 3) + vc * np.exp(1j * 2 * np.pi / 3)) / 3

V1_magn = abs(V1)


folder = 'over-voltage-scenario-new'

grid.solve_power_flow(dss_path='test_grids/IEEE.dss', number_of_seconds=60, starting_second=t-30, save_results=True,
                      folder=folder)

# but make sure that the method returns V in order to check over-voltage
# if there is over-voltage save df, else go to next random iteration
df.to_csv(os.path.join(folder, 'pv_battery.csv'))
df_gen.to_csv(os.path.join(folder, 'generators.csv'))

v0_df = pd.DataFrame(data=abs(V0), index=va.index)
v1_df = pd.DataFrame(data=abs(V1), index=va.index)
v2_df = pd.DataFrame(data=abs(V2), index=va.index)

# v0_df.to_csv(os.path.join(folder, 'V0.csv'))
v1_df.to_csv(os.path.join(folder, 'V1.csv'))
# v2_df.to_csv(os.path.join(folder, 'V2.csv'))





socs = []
cols = []
for k, home in enumerate(grid.homes.keys()):
    # check number of phases
    if grid.homes[home].has_battery:
        cols.append(home)
        socs.append(grid.homes[home].battery['soc'].iloc[t-30:t+30].values)


socs_df = pd.DataFrame(columns=v1_df.index, index=cols, data=socs).T

socs_df.to_csv(os.path.join(folder, 'soc.csv'))
