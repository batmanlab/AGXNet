"""Functions used for PU Learning."""

import numpy as np


def DKW_bound(x, y, t, m, n, delta=0.1, gamma=0.01):
    temp = np.sqrt(np.log(1 / delta) / 2 / n) + np.sqrt(np.log(1 / delta) / 2 / m)
    bound = temp * (1 + gamma) / (y / n)

    estimate = t

    return estimate, t - bound, t + bound

def top_bin_estimator_count(obs_prob_arr, obs_pnu_arr):
    # find the obs.'s original pnu indices
    p_data_idx = obs_pnu_arr == 1
    u_data_idx = obs_pnu_arr == -1

    p_data_prob = obs_prob_arr[p_data_idx]
    u_data_prob = obs_prob_arr[u_data_idx]

    # sorted probabilities
    sorted_p_prob = p_data_prob[np.argsort(p_data_prob)]
    sorted_u_prob = u_data_prob[np.argsort(u_data_prob)]
    sorted_p_prob = sorted_p_prob[::-1]
    sorted_u_prob = sorted_u_prob[::-1]


    num = len(sorted_u_prob)
    i = 0
    j = 0
    upper_cfb = []
    lower_cfb = []
    plot_arr = []
    while (i < num):
        start_interval = sorted_u_prob[i]
        if (i < num - 1 and start_interval > sorted_u_prob[i + 1]):
            pass
        else:
            i += 1
            continue

        while (j < len(sorted_p_prob) and sorted_p_prob[j] >= start_interval):
            j += 1

        if j > 1 and i > 1:
            t = (i) * 1.0 * len(sorted_p_prob) / j / len(sorted_u_prob)
            estimate, lower, upper = DKW_bound(i, j, t, len(sorted_u_prob), len(sorted_p_prob))
            plot_arr.append(estimate)
            upper_cfb.append(upper)
            lower_cfb.append(lower)

        i += 1

    if (len(upper_cfb) != 0):
        mpe_estimate = np.min(upper_cfb)
        idx = np.argmin(upper_cfb)
        alpha_estimate = plot_arr[idx]
    else:
        alpha_estimate = 1.0

    return alpha_estimate
