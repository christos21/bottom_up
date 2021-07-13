import pandas as pd
import numpy as np


def mean_auto_consumption_rate(gen, injected):

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
    numerator = np.minimum(gen, load)
    gamma_s = numerator.sum()/gen.sum()
    gamma_d = numerator.sum()/load.sum()

    return gamma_s, gamma_d


def load_match_index(gen, load, interval='2h'):
    E_gen = gen.resample(interval).sum()
    E_load = load.resample(interval).sum()

    lmi = np.minimum(1, E_gen/E_load)

    return lmi


def loss_of_load_probability(gen, load):
    return 1*(load > gen).sum()/len(gen)


def one_percent_peak_power(p_exchange, period='1s'):
    # quarter hourly power peaks
    p_exchange = abs(p_exchange)
    p_exchange_peaks = p_exchange.resample(period).max()
    p_100 = p_exchange_peaks.values
    p_100.sort()
    # vals = p_100[-max(int(0.01*len(p_100)), 1):] # not mean, keep p_100[-0.01*len(p_100)
    # return vals.mean()
    opp = p_100[int(0.99*len(p_100))]
    return opp


def peaks_above_limit(p_exchange, p_limit=5000):
    return 1*(abs(p_exchange) > abs(p_limit)).sum()/len(p_exchange)*100

# (1*(abs(p_exchange) > abs(p_limit))).sum()/len(p_exchange)*100


def no_grid_interaction_probability(p_exchange, period='15min', limit=0.001):
    # p_exchange = abs(p_exchange)
    e_exchange = p_exchange.resample(period).sum()/(3600*1000)
    return 1*(abs(e_exchange) < abs(limit)).sum()/len(e_exchange)


def capacity_factor(p_exchange, p_cap):
    e = abs(p_exchange).sum()/1000
    return e/(len(p_exchange)*p_cap)


def self_consumption_rate(load, pv, battery_charge=None, battery_discharge=None):
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
    numerator = charge.sum()  # + discharge.sum()
    numerator = numerator/(1000*3600)

    denumerator = len(charge)/(24*3600)*e_max*depth_of_discharge

    bui_charge = numerator/denumerator

    numerator = discharge.sum()
    numerator = numerator/(1000*3600)

    bui_discharge = numerator/denumerator

    return bui_charge, bui_discharge


def substation_reserve_capacity(s_sec, s_substation):
    src = pd.DataFrame(index=s_sec.index, data=1 - abs(s_sec) / s_substation)
    return src


def feeder_loss_to_load_ratio(total_load, losses):
    return losses / (abs(total_load))


def average_feeder_loading_index(line_power, line_length, line_c, total_length):
    afli = pd.DataFrame(index=line_power.index, columns=line_power.columns, data=0)

    for k, line in enumerate(afli.columns):
        afli[line] = line_length[line]*line_power[line]/(total_length*line_c[line])

    return afli.sum(axis=1)
