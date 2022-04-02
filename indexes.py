import pandas as pd
import numpy as np
from utils import sequential_voltage_vectors

"""
This file includes functions for calculating end-user and grid-related indexes.
All indexes are based on the following bibliography:

[1]  J. Pouladi, T. Abedinzadeh, “Performance Evaluation of Distribution Network in Presence of Plug-in Electric 
     Vehicles Through a New Index”, ICSG Istanbul, 2017
[2] M. J. E. Alam, K. M. Muttaqi and D. Sutanto, "An Approach for Online Assessment of Rooftop Solar PV Impacts on 
    Low-Voltage Distribution Networks," in IEEE Transactions on Sustainable Energy, vol. 5, no. 2, pp. 663-672, 
    April 2014. 
[3] F. Vallée, V. Klonari, J. Lobry and O. Durieux, "Study of the combined impact of auto-consumption behaviour and 
    correlation level between prosumers on overvoltage probabilities in low voltage distribution grids", 
    2014 IEEE PES T&D Conference and Exposition, Chicago, IL, 2014
[4] B. Verbruggen and J. Driesen, "Grid Impact Indicators for Active Building Simulations," in IEEE Transactions 
    on Sustainable Energy, vol. 6, no. 1, pp. 43-50, Jan. 2015.
[5] Kalliopi D. Pippi, Theofilos A. Papadopoulos, Georgios C. Kryonidis. "Impact Assessment Framework of PV-BES 
    Systems to Active Distribution Networks"
"""


def mean_auto_consumption_rate(gen, injected):
    """
    Calculates mean auto consumption rate as described in [3].
    The generated and injected power are assumed to have 1 Hz resolution.
    :param gen: pd.Series
    :param injected: pd.Series
    :return: pd.Series (at 15 min. resolution)
    """

    gen_array = gen.values.reshape([-1, 24*60*60]).T
    injected_array = injected.values.reshape([-1, 24*60*60]).T

    gen_df = pd.DataFrame(gen_array, index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))
    injected_df = pd.DataFrame(injected_array, index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq='1s'))

    gen_energy_quarters = gen_df.resample('15min').sum()
    injected_energy_quarters = injected_df.resample('15min').sum()

    gen_generation_index = gen_energy_quarters.index[(gen_energy_quarters > 0).all(axis=1)]

    numerator = gen_energy_quarters - injected_energy_quarters
    numerator = numerator.loc[gen_generation_index]
    denumerator = gen_energy_quarters.loc[gen_generation_index]

    r_auto = (numerator/denumerator).mean(axis=1)

    return r_auto


def cover_factors(gen, load):
    """
    Calculates cover factors as described in [4].
    :param gen: pd.Series
    :param load: pd.Series
    :return: (float, float)
    """
    numerator = np.minimum(gen, load)
    gamma_s = numerator.sum()/gen.sum()
    gamma_d = numerator.sum()/load.sum()

    return gamma_s, gamma_d


def load_match_index(gen, load, interval='2h'):
    """
    Calculates load match index as described in [4].
    Parameter 'interval' is used for down-sampling.
    :param gen: pd.Series
    :param load: pd.Series
    :param interval: str
    :return: pd.Series
    """
    E_gen = gen.resample(interval).sum()
    E_load = load.resample(interval).sum()

    lmi = np.minimum(1, E_gen/E_load)

    return lmi


def loss_of_load_probability(gen, load):
    """
    Calculates loss of load probability as described in [4].
    :param gen: pd.Series
    :param load: pd.Series
    :return: float
    """
    return 1*(load > gen).sum()/len(gen)


def one_percent_peak_power(p_exchange, period='1s'):
    """
    Calculates one percent peak power as described in [4].
    Parameter 'period' is used for down-sampling.
    :param p_exchange: pd.Series
    :param period: str
    :return: float
    """
    p_exchange = abs(p_exchange)
    p_exchange_peaks = p_exchange.resample(period).max()
    p_100 = p_exchange_peaks.values
    p_100.sort()
    opp = p_100[int(0.99*len(p_100))]
    return opp


def peaks_above_limit(p_exchange, p_limit=5000):
    """
    Calculates peaks above limit as described in [4].
    :param p_exchange: pd.Series
    :param p_limit: float | int
    :return: float
    """
    return 1*(abs(p_exchange) > abs(p_limit)).sum()/len(p_exchange)*100


def no_grid_interaction_probability(p_exchange, period='15min', limit=0.001):
    """
    Calculates no grid interaction probability as described in [4].
    Parameter 'period' is used for down-sampling.
    :param p_exchange: pd.Series
    :param period: str
    :param limit: float
    :return: float
    """
    current_freq = (p_exchange.index[1] - p_exchange.index[0]).seconds
    e_exchange = p_exchange.resample(period).sum()/(3600*1000/current_freq)
    return 1*(abs(e_exchange) < abs(limit)).sum()/len(e_exchange)


def capacity_factor(p_exchange, p_cap):
    """
    Calculates capacity factor as described in [4].
    :param p_exchange: pd.Series
    :param p_cap: float | int
    :return: float
    """
    e = abs(p_exchange).sum()/1000
    return e/(len(p_exchange)*p_cap)


def self_consumption_rate(load, pv, battery_charge=None, battery_discharge=None):
    """
    Calculates self consumption rate as described in [5].
    :param load: pd.Series
    :param pv: pd.Series
    :param battery_charge: pd.Series
    :param battery_discharge: pd.Series
    :return: float
    """
    load_battery = load.copy()

    if battery_charge is not None:
        D = battery_charge.sum()
        load_battery += battery_charge
    else:
        D = 0

    production = pv.copy()
    if battery_discharge is not None:
        E = battery_discharge.sum()
        production += battery_discharge
    else:
        E = 0

    load_minus_prod = load - production
    AF = load_minus_prod[load_minus_prod > 0].sum()

    pv_minus_load_battery = pv - load_battery
    B = pv_minus_load_battery[pv_minus_load_battery > 0].sum()

    C = np.minimum(load, pv).sum()

    return (C + E)/(B + C + D)


def self_sufficiency_rate(load, pv, battery_charge=None, battery_discharge=None):
    """
    Calculates self consumption rate as described in [5].
    :param load: pd.Series
    :param pv: pd.Series
    :param battery_charge: pd.Series
    :param battery_discharge: pd.Series
    :return: float
    """
    load_battery = load.copy()

    if battery_charge is not None:
        D = battery_charge.sum()
        load_battery += battery_charge
    else:
        D = 0

    production = pv.copy()
    if battery_discharge is not None:
        E = battery_discharge.sum()
        production += battery_discharge
    else:
        E = 0

    load_minus_prod = load - production
    AF = load_minus_prod[load_minus_prod > 0].sum()

    pv_minus_load_battery = pv - load_battery
    B = pv_minus_load_battery[pv_minus_load_battery > 0].sum()

    C = np.minimum(load, pv).sum()

    return (C + E)/(AF + C + E)


def battery_utilization_index(charge, discharge, e_max, depth_of_discharge):
    """
    Calculates battery utilization index for charging and discharging as described in [5].
    :param charge: pd.Series
    :param discharge: pd.Series
    :param e_max: float
    :param depth_of_discharge: float
    :return: (float, float)
    """
    current_freq = (charge.index[1] - discharge.index[0]).seconds

    numerator = charge.sum()
    numerator = numerator/(1000*3600/current_freq)

    denumerator = len(charge)/(24*3600/current_freq)*e_max*depth_of_discharge

    bui_charge = numerator/denumerator

    numerator = discharge.sum()
    numerator = numerator/(1000*3600/current_freq)

    bui_discharge = numerator/denumerator

    return bui_charge, bui_discharge


def substation_reserve_capacity(s_sec, s_substation):
    """
    Calculates substation reserve capacity as described in [1].
    :param s_sec: pd.DataFrame
    :param s_substation: float
    :return: pd.DataFrame
    """
    src = pd.DataFrame(index=s_sec.index, data=1 - abs(s_sec) / s_substation)
    return src


def feeder_loss_to_load_ratio(total_load, losses):
    """
    Calculates feeder loss to load ratio as described in [1], [2].
    :param total_load: pd.Series
    :param losses: pd.Series
    :return: pd.Series
    """
    fllr = losses.values / abs(total_load.values)
    return pd.Series(data=fllr, index=total_load.index)


def losses_to_load_ratio(total_load, active_losses):
    """
    Calculates losses to load ratio as described in [5].
    :param total_load: pd.Series
    :param active_losses: pd.Series
    :return: pd.Series
    """
    return active_losses.sum() / total_load.sum()


def losses_reduction_index(active_losses, passive_losses):
    """
    Calculates losses reduction index as described in [5].
    :param active_losses: pd.Series
    :param passive_losses: pd.Series
    :return: pd.Series
    """
    return active_losses / passive_losses


def voltage_unbalance_factors(va, vb, vc):
    """
    Calculates voltage unbalance factors as described in [5].
    :param va: pd.DataFrame
    :param vb: pd.DataFrame
    :param vc: pd.DataFrame
    :return: (pd.DataFrame, pd.DataFrame)
    """
    V0, V1, V2 = sequential_voltage_vectors(va, vb, vc)
    V0, V1, V2 = abs(V0), abs(V1), abs(V2)
    VUF0 = V0/V1 * 100
    VUF2 = V2/V1 * 100

    return VUF0, VUF2


def mean_voltage_variance(v, freq=None):
    """
    Calculates mean voltage variance as described in [5].
    :param v: pd.Dataframe
    :return: pd.DataFrame
    """
    current_freq = (pd.Timedelta(v.index[1]) - pd.Timedelta(v.index[0])).seconds

    if freq is None:
        freq = '{}s'.format(current_freq)

    buses = v.columns

    result = {}

    for bus in buses:

        v_values = v[bus].to_numpy().reshape([-1, (24*60*60)//current_freq]).T

        _, days = v_values.shape

        daily_v_df = pd.DataFrame(v_values, index=pd.timedelta_range(start='00:00:00', end='23:59:59',
                                                                     freq='{}s'.format(current_freq)))

        daily_v_df = daily_v_df.resample(freq).mean()

        if days < 2:
            days = 2
            mean_v_per_time_step = pd.Series(index=daily_v_df.index, data=np.nan)
            hourly_mean = daily_v_df.resample('1h').mean()
            mean_v_per_time_step.loc[hourly_mean.index] = hourly_mean.values.reshape(-1)
            mean_v_per_time_step = mean_v_per_time_step.ffill()

            # mean_v_per_time_step = daily_v_df.resample('1h').mean().resample(freq).pad()
        else:

            mean_v_per_time_step = daily_v_df.mean(axis=1)

        sigma_df = pd.DataFrame(data=(daily_v_df.values - mean_v_per_time_step.values.reshape(-1, 1))**2,
                                index=mean_v_per_time_step.index)

        sigma = sigma_df.sum(axis=1)/(days-1)

        result[bus] = sigma

    return pd.DataFrame(result, index=pd.timedelta_range(start='00:00:00', end='23:59:59', freq=freq))


def voltage_level_quantification_index(va, vb, vc, passive_va, passive_vb, passive_vc, r_to_bus, active_nodes=None):
    """
    Calculates voltage level quantification index as described in [5].
    :param va: pd.Dataframe
    :param vb: pd.Dataframe
    :param vc: pd.Dataframe
    :param passive_va: pd.Dataframe
    :param passive_vb: pd.Dataframe
    :param passive_vc: pd.Dataframe
    :param r_to_bus: dict
    :return: float
    """
    # Vnom is not required since it is the same and for VLQI it can be removed?
    # Is R for each part R0?
    # V1 is abs or vector?
    w = r_to_bus

    V0, V1, V2 = sequential_voltage_vectors(va, vb, vc)
    V0, V1, V2 = abs(V0), abs(V1), abs(V2)

    passive_V0, passive_V1, passive_V2 = sequential_voltage_vectors(passive_va, passive_vb, passive_vc)
    passive_V0, passive_V1, passive_V2 = abs(passive_V0), abs(passive_V1), abs(passive_V2)

    if active_nodes:
        active = V1.loc[:, active_nodes].copy()
        passive = passive_V1.loc[:, active_nodes].copy()
    else:
        active = V1.copy()
        passive = passive_V1.copy()

    for col in active.columns:
        if col not in w.keys() or not w[col]:
            continue
        active[col] = active[col] * w[col]
        passive[col] = passive[col] * w[col]

    vlqi = active.sum(axis=1)/passive.sum(axis=1)
    vlqi = vlqi.sum()/len(V0)

    return vlqi


def average_feeding_loading_index(length_to_bus, line_length, line_c, line_power):
    """
    Calculates average feeding loading index as described in [2].
    :param length_to_bus: dict
    :param line_length: dict
    :param line_c: dict
    :param line_power: pd.Dataframe
    :return: pd.DataFrame
    """
    lengths = list(length_to_bus.values())
    # to remove None
    lengths = [l for l in lengths if l]
    L = max(lengths)

    line_c = dict((k.lower(), v) for k, v in line_c.items())

    afli = pd.DataFrame(index=line_power.index, columns=line_power.columns, data=0)

    for k, line in enumerate(afli.columns):
        afli[line] = line_length[line]*line_power[line]/(L*line_c[line])

    return afli.sum(axis=1)

