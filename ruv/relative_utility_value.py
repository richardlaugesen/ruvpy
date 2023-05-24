# Copyright 2023 Richard Laugesen
#
# This file is part of RUV
#
# RUV is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RUV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RUV.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.optimize import minimize_scalar
from pathos.pools import ProcessPool as Pool


# 5 times faster then statsmodels ecdf
def ecdf_numpy(ens, thresholds):
    ens_sorted = np.sort(ens)
    idx = np.searchsorted(ens_sorted, thresholds)
    probs = np.array(np.arange(ens.size + 1))/float(ens.size)  # 3 times fast then linspace
    return 1 - probs[idx]


# Forecast probability of an ensemble within each flow class for single timestep
def calc_likelihoods(ens, thresholds):
    # continuous flow classes
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])

    if np.any(np.isnan(ens)):
        raise ValueError('Cannot calculate likelihood of ensemble with missing values')

    if isinstance(ens, (int, float)) or len(ens) == 1:
        raise ValueError('Likelihood for deterministic forecast (single value) not implemented')

    # finite flow classes
    probs_above = ecdf_numpy(ens, thresholds)
    adjustment = np.roll(probs_above, -1)
    adjustment[-1] = 0.0
    probs_between = np.subtract(probs_above, adjustment)
    return probs_between


# Forecast probability of a series ensemble within each flow class for set of timesteps
# Number of timesteps defined by the prodived obs series
# TODO: why is it NA for timesteps where obs is NA? just a perfomance thing? If so then lets remove it
def all_likelihoods(obs, ensembles, thresholds):
    likelihoods = np.full((obs.shape[0], ensembles.shape[1] if thresholds is None else thresholds.shape[0]), np.nan)

    for t, ob in enumerate(obs):
        if not np.isnan(ob):
            likelihoods[t] = calc_likelihoods(ensembles[t], thresholds)

    return likelihoods


# which flow class is the value in
def realised_threshold(value, thresholds):
    # continuous flow classes
    if thresholds is None:
        return value

    if value < np.min(thresholds):
        raise ValueError('Value is less than smallest threshold')

    # finite flow classes
    vals = np.subtract(value, thresholds)
    idx = np.argmin(vals[vals >= 0.0])

    return thresholds[idx]

 
# 'event frequency' reference distribution for each timestep 
# is simply the obs record with any missing values dropped
def event_freq_ref(obs):
    return np.tile(obs[~np.isnan(obs)], (obs.shape[0], 1))


# convert probablisitic forecast into deterministic according to some 
# decision level defined by a critical probability threshold
def probabilistic_to_deterministic_forecast(fcst_ensemble, decision_level):
    return np.nanquantile(fcst_ensemble, 1 - decision_level, axis=1)


# ex ante expected utility for single timestep
def ex_ante_utility(spend, likelihoods, thresholds, alpha, economic_model, damage_function, utility_function):
    net_expenses = economic_model(thresholds, spend, alpha, damage_function)
    expected_utility = np.sum(np.multiply(likelihoods, utility_function(net_expenses)))
    return expected_utility


# ex post expected utility for single timestep
def ex_post_utility(occured, spend, alpha, economic_model, damage_function, utility_function):
    net_expense = economic_model(occured, spend, alpha, damage_function)
    return utility_function(net_expense)


# Amount to spend for a single timestep
# Fast analytical method for deterministic forecast, pre-calculated likelihood for probabilistic
def find_spend(fcst, likelihoods, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function):
    
    # deterministic
    if isinstance(fcst, (int, float)) or len(fcst) == 1:
        return analytical_spend(realised_threshold(fcst, thresholds), alpha, damage_function)
    
    # probabilistic
    else:
        thresholds = fcst if thresholds is None else thresholds    # Continuous flow decision

        # Find optimial spend amount
        def minimise_this(spend):
            return -ex_ante_utility(spend, likelihoods, thresholds, alpha, economic_model, damage_function, utility_function)        

        return minimize_scalar(minimise_this, method='brent').x  # TODO: experiment with other methods        


# TODO: do all this inplace on an aligned numpy array for efficiency, rather than functionally
def single_timestep(t, ob, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function, fcst, fcst_likelihoods, ref, ref_likelihoods):

    # threshold that occurred
    ob_threshold = realised_threshold(ob, thresholds)

    # perfect
    obs_spends = find_spend(ob, None, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function)
    obs_ex_post = ex_post_utility(ob_threshold, obs_spends, alpha, economic_model, damage_function, utility_function)
    
    # forecast
    fcst_spends = find_spend(fcst, fcst_likelihoods, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function)
    fcst_ex_post = ex_post_utility(ob_threshold, fcst_spends, alpha, economic_model, damage_function, utility_function)

    # reference
    ref_spends = find_spend(ref, ref_likelihoods, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function)
    ref_ex_post = ex_post_utility(ob_threshold, ref_spends, alpha, economic_model, damage_function, utility_function)

    return t, obs_spends, obs_ex_post, fcst_spends, fcst_ex_post, ref_spends, ref_ex_post


# Calculate RUV for a single alpha value by finding spend amounts and utilities for all timesteps
# Timesteps are parallelised over multiple CPU cores
def single_alpha(alpha, obs, fcst, ref, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, cpus, verbose):

    fcst_ex_post = 0.0
    obs_ex_post = 0.0
    ref_ex_post = 0.0
    fcst_spends = np.full(obs.shape[0], np.nan)
    obs_spends = np.full(obs.shape[0], np.nan)
    ref_spends = np.full(obs.shape[0], np.nan)
    fcst_ex_post = np.full(obs.shape[0], np.nan)
    obs_ex_post = np.full(obs.shape[0], np.nan)
    ref_ex_post = np.full(obs.shape[0], np.nan)

    args = []
    for t, ob in enumerate(obs):
        if not np.isnan(ob):
            args.append([t, ob, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function, fcst[t], fcst_likelihoods[t], ref[t], ref_likelihoods[t]])
    args = list(map(list, zip(*args)))

    with Pool(nodes=cpus) as pool:
        results = pool.map(single_timestep, *args)

    for completed_result in results:
        t = completed_result[0]
        obs_spends[t] = completed_result[1]
        obs_ex_post[t] = completed_result[2]
        fcst_spends[t] = completed_result[3]
        fcst_ex_post[t] = completed_result[4]
        ref_spends[t] = completed_result[5]
        ref_ex_post[t] = completed_result[6]

    fcst_avg_ex_post = np.nanmean(fcst_ex_post)
    obs_avg_ex_post = np.nanmean(obs_ex_post)
    ref_avg_ex_post = np.nanmean(ref_ex_post)

    ruv = (ref_avg_ex_post - fcst_avg_ex_post) / (ref_avg_ex_post - obs_avg_ex_post)

    if verbose:
        print('Alpha: %.3f   RUV: %.2f' % (alpha, ruv))
        
    return [ruv, fcst_avg_ex_post, obs_avg_ex_post, ref_avg_ex_post, fcst_spends, obs_spends, ref_spends, fcst_ex_post, obs_ex_post, ref_ex_post]


# Relative Economic Model metric
#   thresholds = None means to run for continuous flow
#   ref = None means to use 'event frequency'
def calc_ruv(obs, fcst, ref, thresholds, alphas, economic_model, analytical_spend, damage_function, utility_function, cpus=1, fcst_threshold_equals_alpha=False, verbose=False):

    # handle event frequency reference
    if ref is None:
        using_event_freq_ref = True
        ref = event_freq_ref(obs)
    else:
        using_event_freq_ref = False

    # pre-initialise data structures to return
    shape = (alphas.shape[0])
    ruvs = np.full(shape, np.nan)
    fcst_avg_ex_post = np.full(shape, np.nan)
    obs_avg_ex_post = np.full(shape, np.nan)
    ref_avg_ex_post = np.full(shape, np.nan)
    fcst_spends = {}
    obs_spends = {}
    ref_spends = {}
    fcst_ex_post = {}
    obs_ex_post = {}
    ref_ex_post = {}

    fcst_likelihoods = all_likelihoods(obs, fcst, thresholds)
    ref_likelihoods = all_likelihoods(obs, ref, thresholds)

    # calc RUV for each alpha value (over multiple CPU cores)
    for a, alpha in enumerate(alphas):
        using_fcst = fcst
        using_ref = ref

        # convert prob forecast to det using critical forecast threshold set to alpha
        if fcst_threshold_equals_alpha:
            using_fcst = probabilistic_to_deterministic_forecast(fcst, alpha)
            using_fcst = np.reshape(using_fcst, (1, len(using_fcst)))[0]  

            # only adjust the ref distribution if its not the event frequency
            if not using_event_freq_ref:
                using_ref = probabilistic_to_deterministic_forecast(ref, alpha)
                using_ref = np.reshape(using_ref, (1, len(using_ref)))[0]  

        single_alpha_result = single_alpha(alpha, obs, using_fcst, using_ref, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, cpus, verbose)

        ruvs[a] = single_alpha_result[0]
        fcst_avg_ex_post[a] = single_alpha_result[1]
        obs_avg_ex_post[a] = single_alpha_result[2]
        ref_avg_ex_post[a] = single_alpha_result[3]
        fcst_spends[a] = single_alpha_result[4]
        obs_spends[a] = single_alpha_result[5]
        ref_spends[a] = single_alpha_result[6]
        fcst_ex_post[a] = single_alpha_result[7]
        obs_ex_post[a] = single_alpha_result[8]
        ref_ex_post[a] = single_alpha_result[9]

    result = {
        'ruv': ruvs,
        'fcst_avg_ex_post': fcst_avg_ex_post,
        'obs_avg_ex_post': obs_avg_ex_post,
        'ref_avg_ex_post': ref_avg_ex_post,
        'fcst_spends': fcst_spends,
        'obs_spends': obs_spends,
        'ref_spends': ref_spends,
        'fcst_likelihoods': fcst_likelihoods,
        'ref_likelihoods': ref_likelihoods
    }

    return result


def relative_utility_value(obs, fcst, ref, decision_def, step=0.01, alphas=None, threshold_eq_alpha=False, cpus=4, verbose=False):
    alphas = np.arange(step, 1, step) if alphas is None else alphas
    
    damage_fnc_mth = decision_def['damage_function'][0]
    damage_fnc_params = decision_def['damage_function'][1]
    damage_fnc = damage_fnc_mth(damage_fnc_params)

    utility_fnc_mth = decision_def['utility_function'][0]
    utility_fnc_params = decision_def['utility_function'][1]
    utility_fnc = utility_fnc_mth(utility_fnc_params)

    decisions = decision_def['decision_thresholds']
    econ_model, fast_spend = decision_def['economic_model']

    result = calc_ruv(obs, fcst, ref, decisions, alphas, econ_model, fast_spend, damage_fnc, utility_fnc, cpus, threshold_eq_alpha, verbose=verbose)
    
    return result['ruv'], result, alphas
