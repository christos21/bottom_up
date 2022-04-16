import numpy as np


def pv_bess_sizing(p, normalized_pv, single_phase,
                   p_bat_max, p_bat_min, e_bat_max, e_bat_min,
                   single_phase_pv_max_p, three_phase_pv_min_p, three_phase_pv_max_p,
                   a=0.5, C=0.5):

    mean_produced_power = np.mean(normalized_pv)
    mean_consumed_power = p.mean()

    ratio = mean_consumed_power / mean_produced_power
    ratio = np.ceil(ratio)

    if single_phase:
        pv_rated = min(ratio, single_phase_pv_max_p)
    else:
        pv_rated = min(three_phase_pv_max_p, max(ratio, three_phase_pv_min_p))

    # Battery power is half the PV power
    p_bat = np.ceil(a * pv_rated)

    if p_bat < p_bat_min:
        p_bat = 0
    else:
        p_bat = min(p_bat, p_bat_max)

    # Calculate the excess of energy produced by the PV, only during production hours
    energy_excess = np.sum((ratio * 1000 * normalized_pv - p.values) * (normalized_pv > 0)) / (1000 * 3600)

    if energy_excess < e_bat_min:
        e_bat = 0
    else:
        e_bat = min(np.ceil(energy_excess), e_bat_max)

    p_bat = min(C*e_bat, p_bat)

    return pv_rated, p_bat, e_bat
