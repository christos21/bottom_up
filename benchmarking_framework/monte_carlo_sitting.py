import pandas as pd
import numpy as np


def random_sitting(houses, pv_bes_parameters, pv_rated_per_home, e_bat_per_home, p_bat_per_home,
                   soc_init, soc_min, soc_max, ch_eff, dch_eff, t_lpf_bat,
                   probability_of_pv, probability_of_battery_given_pv,
                   available_nodes_for_solar_parks, number_of_solar_parks, pv_power_for_solar_parks):
    """
    This function places PV/BES units and solar parks in random nodes of the DN.
    :return: pd.DataFrame, pd.DataFrame
    """

    # initialize empty df with pvv/bes parameters for all houses
    df = pd.DataFrame(columns=houses, index=pv_bes_parameters)

    # for each home
    for k, home in enumerate(houses):
        # Set randomly PV based on the probability but pv_rated should be > 0
        if np.random.random() < probability_of_pv and pv_rated_per_home[k] > 0:

            df.loc['PV_rated', home] = pv_rated_per_home[k]

            # Set randomly battery based on the probability but the home MUST have PV
            if np.random.random() < probability_of_battery_given_pv and e_bat_per_home[k] > 0 \
                    and p_bat_per_home[k] > 0:
                df.loc['SoC_init', home] = soc_init
                df.loc['SoC_min', home] = soc_min
                df.loc['SoC_max', home] = soc_max
                df.loc['ch_eff', home] = ch_eff
                df.loc['dch_eff', home] = dch_eff
                df.loc['t_lpf_bat', home] = t_lpf_bat
                df.loc['P_max_bat', home] = p_bat_per_home[k]
                df.loc['E_max', home] = e_bat_per_home[k]

    # Add random generators
    gen_buses = np.random.choice(available_nodes_for_solar_parks, number_of_solar_parks, replace=False)
    df_gen = pd.DataFrame(columns=gen_buses, index=['phase_number', 'rated'])
    df_gen.loc['rated'] = pv_power_for_solar_parks

    return df, df_gen
