import os.path
from power_flow import solve_power_flow
from smart_grid import SmartGrid
from utils import pv_profile_generator
import pandas as pd
import numpy as np
from scipy.io import loadmat

# initialize the grid
grid_csv = 'test_cases/grid_IEEE.csv'
simulation_file = 'test_grids/simulation_parameters.csv'

grid = SmartGrid(grid_csv, name='IEEE')
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

line_limits = loadmat('Line_data.mat')
line_name = line_limits['Linename']
line_name = [ln[0][0] for ln in line_name]
line_ampacity = line_limits['ampacity']
line_ampacity = [la[0] for la in line_ampacity]

line_limits = {line_name[k]: line_ampacity[k] for k in range(len(line_name))}


def check_current_limits(i, line_limits):
    lines = i.columns

    for line in lines:
        i_line = i[line]
        if line in line_limits.keys():
            limit = line_limits[line]
        else:
            limit = line_limits[line.upper()]

        if (i_line > limit).any():
            return False

    return True


# read month in order to get pv power and house names
simulation_params = pd.read_csv(simulation_file, index_col=0, header=None, squeeze=True).to_dict()
month = simulation_params['month']
df = pd.read_csv(grid_csv, index_col=0)
cols = df.columns


# rows to be completed for resetting pv and battery
rows = ['SoC_init', 'P_max_bat', 'E_max', 'SoC_min', 'SoC_max', 'PV_rated', 'ch_eff', 'dch_eff', 't_lpf_bat']
df = pd.DataFrame(columns=cols, index=rows)

# fixed values. Is there a reason to change these?
soc_init = 0.1
soc_min = 0.1
soc_max = 0.9
ch_eff = 0.9
dch_eff = 0.9
t_lpf_bat = 100


battery_min_kW = 3
battery_max_kW = 6

battery_min_kWh = 3
battery_max_kWh = 20

# pv profile to get mean daily production
pv_power = pv_profile_generator(month)
mean_daily_pv_power = np.mean(pv_power)

# calculate pv_rated for each home as the mean daily consumption over mean daily production
# Should we change e_max and p_max for battery accordingly?
pv_rated_per_home = []
bat_p_max_per_home = []
bat_e_max_per_home = []

for home in cols:
    home_mean_daily_power = grid.homes[home].P.sum(axis=1).mean()/1000
    ratio = home_mean_daily_power/mean_daily_pv_power

    ratio = np.ceil(ratio)

    if grid.homes[home].single_phase:
        pv_rated = min(ratio, 5)
    else:
        pv_rated = min(20, max(ratio, 5))

    pv_rated_per_home.append(pv_rated)

    battery_p = np.ceil(pv_rated/2)
    if battery_p < battery_min_kW:
        bat_p_max_per_home.append(0)
    else:
        bat_p_max_per_home.append(min(battery_p, battery_max_kW))

    energy_excess = np.sum((ratio*1000*pv_power - grid.homes[home].P.sum(axis=1).values)*(pv_power > 0))/(1000*3600)

    if energy_excess < battery_min_kWh:
        bat_e_max_per_home.append(0)
    else:
        bat_e_max_per_home.append(min(np.ceil(energy_excess), battery_max_kWh))


for i in range(len(bat_p_max_per_home)):
    if bat_p_max_per_home[i] > bat_e_max_per_home[i]/2:
        bat_p_max_per_home[i] = bat_e_max_per_home[i]/2

# rows to change, add e_max and p_max?
changed_rows = ['PV_rated']


# probability for pv and battery (conditional)
probability_of_pv = 0.6
probability_of_battery_given_pv = 0.4

number_of_pvs_in_free_nodes = 8

# free terminal nodes
term_nodes = ['b3', 'b14', 'b16', 'b25', 'b37', 'b38', 'b44', 'b68', 'b72', 'b74', 'b76', 'b78', 'b80', 'b82', 'b84', 'b87',
              'b88', 'b91', 'b93', 'b96', 'b100', 'b102', 'b106', 'b109', 'b111', 'b115', 'b117', 'b121', 'b123',
              'b126', 'b128', 'b129', 'b133', 'b139', 'b140', 'b142', 'b144', 'b145', 'b160', 'b176', 'b178', 'b180',
              'b182', 'b184', 'b187', 'b189', 'b191', 'b195', 'b197', 'b201', 'b202', 'b203']


# here I should include a loop to check random scenarios
# if there is no over-voltage for e.g. 100 random scenarios maybe I should increase/decrease the probabilities?
attempts = 0
while attempts < 100:
    attempts += 1
    print('Attempt number {}'.format(attempts))
    phases = []
    pvs = []
    battery = []
    # re-initialize the df inside the loop
    df = pd.DataFrame(columns=cols, index=rows)

    # for each home
    for k, home in enumerate(cols):
        # check number of phases
        if grid.homes[home].single_phase:
            phases.append(1)
            # probability 40% to have PV but pv_rated should be > 0
            if np.random.random() < probability_of_pv and pv_rated_per_home[k] > 0:
                pvs.append(1)
                df.loc['PV_rated', home] = pv_rated_per_home[k]

                # probability of having battery is 50% but it MUST have PV
                if np.random.random() < probability_of_battery_given_pv and bat_e_max_per_home[k] > 0 \
                        and bat_p_max_per_home[k] > 0:

                    battery.append(1)

                    # these are fixed for the time being
                    df.loc['SoC_init', home] = soc_init
                    df.loc['SoC_min', home] = soc_min
                    df.loc['SoC_max', home] = soc_max
                    df.loc['ch_eff', home] = ch_eff
                    df.loc['dch_eff', home] = dch_eff
                    df.loc['t_lpf_bat', home] = t_lpf_bat

                    # also fixed, but probably they will change per house
                    df.loc['P_max_bat', home] = bat_p_max_per_home[k]  #5
                    df.loc['E_max', home] = bat_e_max_per_home[k]  #10
                else:
                    battery.append(0)
            else:
                pvs.append(0)
                battery.append(0)

        else:
            # for three phase, the same process
            phases.append(3)
            if np.random.random() < probability_of_pv and pv_rated_per_home[k] > 0:
                pvs.append(1)
                df.loc['PV_rated', home] = pv_rated_per_home[k]
                if np.random.random() < probability_of_battery_given_pv and bat_e_max_per_home[k] > 0 \
                        and bat_p_max_per_home[k] > 0:
                    battery.append(1)

                    df.loc['SoC_init', home] = soc_init
                    df.loc['SoC_min', home] = soc_min
                    df.loc['SoC_max', home] = soc_max
                    df.loc['ch_eff', home] = ch_eff
                    df.loc['dch_eff', home] = dch_eff
                    df.loc['t_lpf_bat', home] = t_lpf_bat

                    df.loc['P_max_bat', home] = bat_p_max_per_home[k]  #5
                    df.loc['E_max', home] = bat_e_max_per_home[k]  #10
                else:
                    battery.append(0)

            else:
                pvs.append(0)
                battery.append(0)

    grid.reset_pv_and_battery(df)

    # add random generators
    gen_buses = np.random.choice(term_nodes, number_of_pvs_in_free_nodes, replace=False)
    df_gen = pd.DataFrame(columns=gen_buses, index=['phase_number', 'rated'])
    df_gen.loc['rated'] = 20

    grid.set_generators(df_gen)

    # for over-voltage check when argmax(grid.P) Q: sum for all phases??
    t = np.argmax(grid.P.sum(axis=1))
    # then
    # grid.solve_power_flow(dss_path, number_of_seconds=60, starting_second=argmax, save_results=False, folder=None)

    va, vb, vc, ia, ib, ic, _ = solve_power_flow(grid, dss_path='test_grids/IEEE.dss', number_of_seconds=60, starting_second=t-30,
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

    print('Maximum V1 {}'.format(np.max(V1_magn.values)))

    if (V1_magn > 1.115).sum().sum() and not (check_current_limits(ia, line_limits) and \
            check_current_limits(ib, line_limits) and check_current_limits(ic, line_limits)):
        break

folder = 'over_voltage_and_thermal_limits_scenario'

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



