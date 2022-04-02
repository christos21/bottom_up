from data_config import PROFILES_PATH

import os
import pandas as pd
import numpy as np
import re
import scipy.io


day_type = {'Mon': 'WD',
            'Tue': 'WD',
            'Wed': 'WD',
            'Thu': 'WD',
            'Fri': 'WD',
            'Sat': 'NWD',
            'Sun': 'NWD'}

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

RATED_POWER = {'AirCon': 1400, 'BEV': 6600, 'Dishwasher': 2100, 'Dryer': 2300, 'eBike': 200, 'Fridge': 170,
               'HairDryer': 2100, 'HeatPump': 2600, 'Iron': 2200, 'LED': 60, 'Lighting': 200,
               'PC': 500, 'Range': 2200, 'Toaster': 700, 'TV': 108, 'WashingMachine': 2300, 'WaterHeater': 4000}

PHASES = {0: 'a', 1: 'b', 2: 'c'}
PHASES_INV = {'a': 0, 'b': 1, 'c': 2}


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def phase_allocation(appliances):
    """
    Function for phase allocation as presented in the manual.
    :param appliances: [str]
    :return: dict {str: str} mapping an appliance to its phase
    """
    appliance_power = [RATED_POWER[appliance] for appliance in appliances]
    # sort
    sort_index = np.argsort(appliance_power)[::-1]
    appliances = [appliances[i] for i in sort_index]

    p_per_phase = np.zeros(3)
    appliance_to_phase = {}
    for appliance in appliances:
        min_arg = p_per_phase.argmin()
        p_per_phase[min_arg] += RATED_POWER[appliance]
        appliance_to_phase[appliance] = PHASES[min_arg]
    return appliance_to_phase


# def pv_profile_generator(month):
#     """
#     Returns the normalized power produced by a solar panel in a specific month.
#     :param month: int
#     :return: np.array
#     """
#     file = os.path.join(PROFILES_PATH, 'PV', 'monthly_profiles.csv')
#     df = pd.read_csv(file, index_col=0)
#     profile = df[df.columns[month]]
#     return profile.values


def pv_profile_generator(month, profile=0):
    """
    Returns the normalized power produced by a solar panel in a specific month.
    :param month: int
    :return: np.array
    """
    file = os.path.join(PROFILES_PATH, 'PV', '{}.csv'.format(month+1))
    df = pd.read_csv(file, index_col=0)
    profile = df[df.columns[profile]]
    return profile.values


def check_for_consecutive_days(params):
    """
    Given a dictionary with the simulation parameters, this function checks if the days for simulation are
    consecutive or not. Furthermore, the days are considered Monday to Sunday, meaning that in case the input days
    are Monday, Tuesday, Sunday the simulation will be applied from Sunday to Tuesday in order to have consecutive days.
    :param params: dict
    :return: bool, [str]
    """
    binary = [params[day] for day in DAYS]

    if sum(binary) == 7:
        return True, DAYS

    binary.insert(0, binary[-1])
    binary = np.array(binary)

    diffs = binary[1:] - binary[:-1]

    raises = list(diffs).count(1)
    drops = list(diffs).count(-1)
    if raises != 1 or drops != 1:
        return False, None

    first_day = np.where(diffs == 1)[0][0]
    last_day = np.where(diffs == -1)[0][0]

    if last_day > first_day:
        return True, DAYS[first_day:last_day]

    days = DAYS[first_day:]
    days.extend(DAYS[:last_day])

    return True, days


def battery_function(p, pv, soc_init, soc_min, soc_max, e_max, ch_eff, dch_eff, p_max_bat, t_lpf_bat):
    """
    Function for implementation of battery storage system as described in the manual.
    :param p: pd.Series
    :param pv: pd.Series
    :param soc_init: float
    :param soc_min: float
    :param soc_max: float
    :param e_max: float
    :param ch_eff: float
    :param dch_eff: float
    :param p_max_bat: float
    :param t_lpf_bat: int
    :return: pd.DataFrame
    """

    time_steps = len(p)
    SoC = np.zeros(time_steps + 1)
    Pch_bat = np.zeros(time_steps)
    Pdch_bat = np.zeros(time_steps)
    Grid_power = np.zeros(time_steps)

    index = p.index
    p = p.values
    pv = pv.values

    SoC[0] = soc_init

    p_bat_prev = 0
    p_bat_temp = 0
    p_bat_temp_prev = 0

    for t in range(time_steps):
        if pv[t] > p[t]:
            # Battery charge algorithm
            # Check if battery is fully charged
            if SoC[t] < soc_max:
                # Calculate required power to reach SoC_max in one second
                p_cap_bat = e_max * ((soc_max - SoC[t]) / ch_eff) * 3600
                p_bat_temp = min(p_cap_bat, p_max_bat, pv[t] - p[t])
                # Apply LPF
                Pch_bat[t] = (p_bat_temp + p_bat_temp_prev) / (2*t_lpf_bat+1) + \
                             ((2 * t_lpf_bat - 1) / (2 * t_lpf_bat + 1)) * p_bat_prev

                p_bat_prev = Pch_bat[t]
                p_bat_temp_prev = p_bat_temp
            else:
                # No charge can be applied
                Pch_bat[t] = 0
                p_bat_prev = 0
                p_bat_temp_prev = 0

            # Discharge is disabled
            Pdch_bat[t] = 0
            # Calculate final grid power
            Grid_power[t] = pv[t] - p[t] - Pch_bat[t]
            # New SoC of the battery
            SoC[t+1] = SoC[t] + Pch_bat[t] * ch_eff/(3600*e_max)

        else:
            # Battery discharge algorithm
            # Check if battery is fully discharged
            if SoC[t] > soc_min:
                # Calculate required power to reach SoC_min in one second
                p_cap_bat = e_max * (SoC[t] - soc_min) * dch_eff * 3600
                # Select the minimum value among Pmax_bat, the above calculated power, and Load-PV_power
                p_bat_temp = min(p_cap_bat, p_max_bat, p[t] - pv[t])
                # Apply LPF
                Pdch_bat[t] = (p_bat_temp + p_bat_temp_prev) / (2 * t_lpf_bat + 1) + \
                              ((2 * t_lpf_bat - 1)/(2 * t_lpf_bat + 1)) * p_bat_prev

                p_bat_prev = Pdch_bat[t]
                p_bat_temp_prev = p_bat_temp
            else:
                # No discharge can be applied
                Pdch_bat[t] = 0
                p_bat_prev = 0
                p_bat_temp_prev = 0

            # Charge is disabled
            Pch_bat[t] = 0
            # Calculate final grid power
            Grid_power[t] = pv[t] - p[t] + Pdch_bat[t]
            # New SoC of the battery
            SoC[t+1] = SoC[t] - (Pdch_bat[t]/dch_eff)/(3600*e_max)

    df = pd.DataFrame(index=index, columns=['soc', 'p_ch_bat', 'p_dch_bat', 'grid_power'], data=0)
    df.soc = SoC[:-1]
    df.p_ch_bat = Pch_bat
    df.p_dch_bat = Pdch_bat
    df.grid_power = Grid_power

    return df


def check_current_limits(lines_current, line_limits):
    """
    Checks for thermal problems in lines.
    The inputs are a dataframe with current for each line and a dictionary with current limit for each line.
    If there is at least a single line with current exceeding the corresponding limit, the function returns 'False',
    else returns 'True'.

    Columns -> lines
    Rows -> time instants

    :param lines_current: pd.DataFrame
    :param line_limits: dict
    :return: bool
    """
    # get all line names
    lines = lines_current.columns

    for line in lines:
        i_line = lines_current[line]
        if line in line_limits.keys():
            limit = line_limits[line]
        else:
            limit = line_limits[line.upper()]

        if (i_line > limit).any():
            return False

    return True



def sequential_voltage_vectors(va, vb, vc):
    """
    Calculates positive, zero and negative sequence voltages.
    TODO: Remove constants and add as parameters

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


def houses_timeseries_to_mat(grid, path_to_save):
    """
    Stores the active and reactive power from each home as well as the PV normalized power in a .mat file.
    :param grid:
    :param path_to_save:
    :return:
    """
    load_loc = []
    phase_con = []
    Pload = []
    Qload = []
    PQloadname = []

    pv_power = pv_profile_generator(grid.month)

    for home in grid.homes.keys():
        h = grid.homes[home]
        load_loc.append(int(home[1:]))
        PQloadname.append(home)
        if h.single_phase:
            phase_con.append(h.selected_phase)
            Pload.append(h.P[h.selected_phase].values)
            Qload.append(h.Q[h.selected_phase].values)
        else:
            phase_con.append('abc')
            for phase in h.P.columns:
                Pload.append(h.P[phase].values)
                Qload.append(h.Q[phase].values)

    load_loc = np.array(load_loc).reshape(-1)
    phase_con = np.array(phase_con)
    Pload = np.array(Pload).T
    Qload = np.array(Qload).T
    PQloadname = np.array(PQloadname)

    dict = {'Load_loc': load_loc,
            'phase_con': phase_con,
            'Pload': Pload,
            'Qload': Qload,
            'PQLoadname': PQloadname,
            'pv': pv_power.T}

    scipy.io.savemat(path_to_save, dict)  # 'bottom_up_data.mat'


def calculate_power_factor(P, Q):
    S = (P ** 2 + Q ** 2)**(1/2)
    return P/S


def mean_power_factor_per_second(grid, weighted_mean=False):

    result = {}

    for home_name in grid.homes.keys():

        home = grid.homes[home_name]

        p_values = home.P.sum(axis=1).to_numpy().reshape([-1, (24*60*60)]).T
        q_values = home.Q.sum(axis=1).to_numpy().reshape([-1, (24 * 60 * 60)]).T

        daily_pf_df = pd.DataFrame(calculate_power_factor(p_values, q_values),
                                   index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

        mean_household_pf_per_second = daily_pf_df.mean(axis=1)

        result[home_name] = mean_household_pf_per_second

    pf_for_all_houses = pd.DataFrame(result, index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

    if not weighted_mean:
        mean_pf_over_all_houses = pf_for_all_houses.mean(axis=1)
    else:
        p_cap_sum = 0
        for home_name in grid.homes.keys():
            pf_for_all_houses[home_name] *= grid.homes[home_name].p_cap
            p_cap_sum += grid.homes[home_name].p_cap

        mean_pf_over_all_houses = pf_for_all_houses.sum(axis=1)/p_cap_sum

    return mean_pf_over_all_houses



def mean_power_factor_for_week(grid, length_to_bus=None, weighted_mean=False, weighted_mean_with_length=False):

    result = {}

    for home_name in grid.homes.keys():

        home = grid.homes[home_name]

        p_values = home.P.sum(axis=1).to_numpy()
        q_values = home.Q.sum(axis=1).to_numpy()

        weekly_pf_df = pd.Series(calculate_power_factor(p_values, q_values),
                                   index=home.P.index)

        result[home_name] = weekly_pf_df

    pf_for_all_houses = pd.DataFrame(result, index=result[list(result.keys())[0]].index)

    if weighted_mean:
        p_cap_sum = 0
        for home_name in grid.homes.keys():
            pf_for_all_houses[home_name] *= grid.homes[home_name].p_cap
            p_cap_sum += grid.homes[home_name].p_cap

        mean_pf_over_all_houses = pf_for_all_houses.sum(axis=1)/p_cap_sum

    elif weighted_mean_with_length and length_to_bus:
        length_sum = 0
        for home_name in grid.homes.keys():
            pf_for_all_houses[home_name] *= length_to_bus[home_name]
            length_sum += length_to_bus[home_name]

        mean_pf_over_all_houses = pf_for_all_houses.sum(axis=1) / length_sum
    else:
        mean_pf_over_all_houses = pf_for_all_houses.mean(axis=1)

    return mean_pf_over_all_houses
