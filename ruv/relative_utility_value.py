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
    if thresholds is None:  # continuous flow classes
        return np.full(ens.shape, 1/ens.shape[0])

    if np.any(np.isnan(ens)):
        raise ValueError('Cannot calculate likelihood with missing values')

    if isinstance(ens, (int, float)) or len(ens) == 1:  # deterministic
        idx = np.where(thresholds == realised_threshold(ens, thresholds))
        probs_between = np.zeros(len(thresholds))
        probs_between[idx] = 1

    else:   # probabilisitc
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
    if thresholds is None:  # continuous
        return value

    if value < np.min(thresholds):
        raise ValueError('Value is less than smallest threshold')

    vals = np.subtract(value, thresholds)
    idx = np.argmin(vals[vals >= 0.0])
    return thresholds[idx]

 
# 'event frequency' reference distribution for each timestep 
# is simply the obs record with any missing values dropped
def generate_event_freq_ref(obs):
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
    if isinstance(fcst, (int, float)) or len(fcst) == 1:    # deterministric
        spend_amount = analytical_spend(realised_threshold(fcst, thresholds), alpha, damage_function)       # TODO: test this is actually faster now I have deterministic likelihoods implemented
    
    else:   # probabilistic
        thresholds = fcst if thresholds is None else thresholds    # Continuous flow decision
        def minimise_this(spend):
            return -ex_ante_utility(spend, likelihoods, thresholds, alpha, economic_model, damage_function, utility_function)        
        spend_amount = minimize_scalar(minimise_this, method='brent').x  # TODO: experiment with other methods        

    return spend_amount


def single_timestep(t, ob, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function, fcst, fcst_likelihoods, ref, ref_likelihoods):
    ob_threshold = realised_threshold(ob, thresholds)
    obs_spends = find_spend(ob, None, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function)
    obs_ex_post = ex_post_utility(ob_threshold, obs_spends, alpha, economic_model, damage_function, utility_function)
    fcst_spends = find_spend(fcst, fcst_likelihoods, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function)
    fcst_ex_post = ex_post_utility(ob_threshold, fcst_spends, alpha, economic_model, damage_function, utility_function)
    ref_spends = find_spend(ref, ref_likelihoods, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function)
    ref_ex_post = ex_post_utility(ob_threshold, ref_spends, alpha, economic_model, damage_function, utility_function)

    result = {
        't': t,
        'obs_spends': obs_spends,
        'obs_ex_post': obs_ex_post,
        'fcst_spends': fcst_spends,
        'fcst_ex_post': fcst_ex_post,
        'ref_spends': ref_spends,
        'ref_ex_post': ref_ex_post
    }
    return result


# Calculate RUV for a single alpha value by finding spend amounts and utilities for all timesteps
# Timesteps are parallelised over multiple CPU cores
def multiple_timesteps(alpha, obs, fcst, ref, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, cpus, verbose):
    fcst_spends, obs_spends, ref_spends, fcst_ex_post, obs_ex_post, ref_ex_post = np.full((6, obs.shape[0]), np.nan)

    args = []
    for t, ob in enumerate(obs):
        if not np.isnan(ob):
            args.append([t, ob, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function, fcst[t], fcst_likelihoods[t], ref[t], ref_likelihoods[t]])
    args = list(map(list, zip(*args)))

    with Pool(nodes=cpus) as pool:
        results = pool.map(single_timestep, *args)  # TODO: try the different map functions to see which is fastest

    for result in results:
        t = result['t']
        obs_spends[t] = result['obs_spends']
        obs_ex_post[t] = result['obs_ex_post']
        fcst_spends[t] = result['fcst_spends']
        fcst_ex_post[t] = result['fcst_ex_post']
        ref_spends[t] = result['ref_spends']
        ref_ex_post[t] = result['ref_ex_post']

    fcst_avg_ex_post = np.nanmean(fcst_ex_post)
    obs_avg_ex_post = np.nanmean(obs_ex_post)
    ref_avg_ex_post = np.nanmean(ref_ex_post)
    ruv = (ref_avg_ex_post - fcst_avg_ex_post) / (ref_avg_ex_post - obs_avg_ex_post)

    if verbose:
        print('Alpha: %.3f   RUV: %.2f' % (alpha, ruv))

    result = {
        'ruv': ruv,
        'fcst_avg_ex_post': fcst_avg_ex_post,
        'obs_avg_ex_post': obs_avg_ex_post,
        'ref_avg_ex_post': ref_avg_ex_post,
        'fcst_spends': fcst_spends,
        'obs_spends': obs_spends,
        'ref_spends': ref_spends,
        'fcst_ex_post': fcst_ex_post,
        'obs_ex_post': obs_ex_post,
        'ref_ex_post': ref_ex_post,
        'alpha': alpha
    }
    return result


# TODO: get rid of these super long function signatures
def multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, crit_prob_eq_alpha, cpus, verbose):
    ruvs, fcst_avg_ex_post, obs_avg_ex_post, ref_avg_ex_post = np.full((4, alphas.shape[0]), np.nan)
    fcst_spends = {}; obs_spends = {}; ref_spends = {}; fcst_ex_post = {}; obs_ex_post = {}; ref_ex_post = {}

    for a, alpha in enumerate(alphas):

        # Use "critical probability threshold equals alpha" method
        curr_fcsts = fcsts; curr_refs = refs
        if crit_prob_eq_alpha:
            curr_fcsts = probabilistic_to_deterministic_forecast(fcsts, alpha)
            curr_refs = probabilistic_to_deterministic_forecast(refs, alpha)
            fcst_likelihoods = all_likelihoods(obs, curr_fcsts, thresholds)
            ref_likelihoods = all_likelihoods(obs, curr_refs, thresholds)

        # calculate results
        results = multiple_timesteps(alpha, obs, curr_fcsts, curr_refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, cpus, verbose)
        ruvs[a] = results['ruv']
        fcst_avg_ex_post[a] = results['fcst_avg_ex_post']
        obs_avg_ex_post[a] = results['obs_avg_ex_post']
        ref_avg_ex_post[a] = results['ref_avg_ex_post']
        fcst_spends[a] = results['fcst_spends']
        obs_spends[a] = results['obs_spends']
        ref_spends[a] = results['ref_spends']
        fcst_ex_post[a] = results['fcst_ex_post']
        obs_ex_post[a] = results['obs_ex_post']
        ref_ex_post[a] = results['ref_ex_post']

    result = {
        'ruv': ruvs,
        'fcst_avg_ex_post': fcst_avg_ex_post,
        'obs_avg_ex_post': obs_avg_ex_post,
        'ref_avg_ex_post': ref_avg_ex_post,
        'fcst_spends': fcst_spends,
        'obs_spends': obs_spends,
        'ref_spends': ref_spends,
        'fcst_likelihoods': fcst_likelihoods,
        'ref_likelihoods': ref_likelihoods,
        'alphas': alphas
    }
    return result


# Relative Economic Model metric
#   thresholds = None means to run for continuous flow
#   ref = None means to use 'event frequency'
def calc_ruv(obs, fcsts, refs, thresholds, alphas, economic_model, analytical_spend, damage_function, utility_function, cpus, crit_prob_eq_alpha, verbose):

    if refs is None:
        refs = event_freq_ref(obs)
        event_freq_ref = True
    else:
        event_freq_ref = False
    
    fcst_likelihoods = all_likelihoods(obs, fcsts, thresholds)
    ref_likelihoods = all_likelihoods(obs, refs, thresholds)

    results = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, cpus, verbose, crit_prob_eq_alpha, event_freq_ref)

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


def relative_utility_value(obs, fcst, ref, decision_def, step=0.01, alphas=None, crit_prob_eq_alpha=False, cpus=4, verbose=False):
    alphas = np.arange(step, 1, step) if alphas is None else alphas
    
    damage_fnc_mth = decision_def['damage_function'][0]
    damage_fnc_params = decision_def['damage_function'][1]
    damage_fnc = damage_fnc_mth(damage_fnc_params)

    utility_fnc_mth = decision_def['utility_function'][0]
    utility_fnc_params = decision_def['utility_function'][1]
    utility_fnc = utility_fnc_mth(utility_fnc_params)

    decisions = decision_def['decision_thresholds']     # TODO: define the crit_prob_eq_alpha method by setting these to something? None for continuou, nan for crit_prob_eq_alpha?
    econ_model, fast_spend = decision_def['economic_model']

    result = calc_ruv(obs, fcst, ref, decisions, alphas, econ_model, fast_spend, damage_fnc, utility_fnc, cpus, crit_prob_eq_alpha, verbose=verbose)
    
    return result['ruv'], result, alphas
