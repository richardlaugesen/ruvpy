# Copyright 2023 Richard Laugesen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.optimize import minimize_scalar
from pathos.pools import ProcessPool as Pool
from scipy.stats import bootstrap
import time


# 5 times faster then statsmodels ecdf
def ecdf(ens, thresholds):
    ens_sorted = np.sort(ens)
    idx = np.searchsorted(ens_sorted, thresholds)
    probs = np.arange(ens.size + 1)/float(ens.size)  # 3 times fast then linspace
    return 1 - probs[idx]


# Forecast probability of an ensemble within each flow class for single timestep
def calc_likelihoods(ens, thresholds):
    if ens is None:
        raise ValueError('Ensemble cannot be None, use generate_event_freq_ref() if using event frequency as reference forecast')

    if thresholds is None:  # continuous flow classes
        return np.full(ens.shape, 1/ens.shape[0])

    if np.any(np.isnan(ens)):
        raise ValueError('Cannot calculate likelihood with missing values')

    if isinstance(ens, (int, float)) or len(ens) == 1:  # deterministic
        idx = np.where(thresholds == realised_threshold(ens, thresholds))
        probs_between = np.zeros(len(thresholds))
        probs_between[idx] = 1

    else:   # probabilisitc
        probs_above = ecdf(ens, thresholds)
        adjustment = np.roll(probs_above, -1)
        adjustment[-1] = 0.0
        probs_between = np.subtract(probs_above, adjustment)

    return probs_between


# Forecast probability of a series ensemble within each flow class for set of timesteps
# Number of timesteps defined by the prodived obs series
# TODO: why is it NA for timesteps where obs is NA? just a perfomance thing? If so then lets remove it
# TODO: need test for this new function
def all_likelihoods(obs, ensembles, thresholds):
    if ensembles is None:
        raise ValueError('Ensemble cannot be None, use generate_event_freq_ref() if using event frequency as reference forecast')

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
    if fcst_ensemble is None:   # handle edge case when using event frequency reference distribution
        return None
    else:
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

    return {
        't': t,
        'obs_spends': obs_spends,
        'obs_ex_post': obs_ex_post,
        'fcst_spends': fcst_spends,
        'fcst_ex_post': fcst_ex_post,
        'ref_spends': ref_spends,
        'ref_ex_post': ref_ex_post
    }


# Calculate RUV for a single alpha value by finding spend amounts and utilities for all timesteps
# Timesteps are parallelised over multiple CPU cores
def multiple_timesteps(alpha, obs, fcst, ref, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, parallel_nodes, verbose=False):
    fcst_spends, obs_spends, ref_spends, fcst_ex_post, obs_ex_post, ref_ex_post = np.full((6, obs.shape[0]), np.nan)

    args = []
    for t, ob in enumerate(obs):
        if not np.isnan(ob):
            args.append([t, ob, thresholds, alpha, economic_model, analytical_spend, damage_function, utility_function, fcst[t], fcst_likelihoods[t], ref[t], ref_likelihoods[t]])
    args = list(map(list, zip(*args)))

    with Pool(nodes=parallel_nodes) as pool:
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

    # TODO: replace this with a logger
    if verbose:
        print('Alpha: %.3f   RUV: %.2f' % (alpha, ruv))

    return {
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
        'fcst_likelihoods': fcst_likelihoods,
        'ref_likelihoods': ref_likelihoods
    }


def multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, decision_method, parallel_nodes, verbose=False):
    ruvs, fcst_avg_ex_post, obs_avg_ex_post, ref_avg_ex_post = np.full((4, alphas.shape[0]), np.nan)
    fcst_spends = {}; obs_spends = {}; ref_spends = {}; fcst_ex_post = {}; obs_ex_post = {}; ref_ex_post = {}

    if decision_method == 'critical_probability_threshold_max_value':
        raise NotImplementedError('critical_probability_threshold_max_value method is not implemented yet')

    for a, alpha in enumerate(alphas):

        # Use "critical probability threshold equals alpha" method
        curr_fcsts = fcsts; curr_refs = refs
        if decision_method == 'critical_probability_threshold_equals_alpha':
            curr_fcsts = probabilistic_to_deterministic_forecast(fcsts, alpha)
            curr_refs = probabilistic_to_deterministic_forecast(refs, alpha)
            fcst_likelihoods = all_likelihoods(obs, curr_fcsts, thresholds)
            ref_likelihoods = all_likelihoods(obs, curr_refs, thresholds)

        # calculate RUV and store results
        results = multiple_timesteps(alpha, obs, curr_fcsts, curr_refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_function, utility_function, parallel_nodes, verbose)
        
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

    return {
        'ruv': ruvs,
        'fcst_avg_ex_post': fcst_avg_ex_post,
        'obs_avg_ex_post': obs_avg_ex_post,
        'ref_avg_ex_post': ref_avg_ex_post,
        'fcst_spends': fcst_spends,
        'obs_spends': obs_spends,
        'ref_spends': ref_spends,
        'fcst_ex_post': fcst_ex_post,
        'obs_ex_post': obs_ex_post,
        'ref_ex_post': ref_ex_post,        
        'fcst_likelihoods': fcst_likelihoods,
        'ref_likelihoods': ref_likelihoods
    }


# TODO: get rid of these super long function signatures, could use Data Classes from Py 3.7

#   thresholds = None means to run for continuous flow
#   refs = None means to use 'event frequency'
def relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=4, verbose=False):

    alphas = decision_definition['alphas']

    # construct damage, utility, and economic functions
    damage_fnc_mth = decision_definition['damage_function'][0]
    damage_fnc_params = decision_definition['damage_function'][1]
    damage_fnc = damage_fnc_mth(damage_fnc_params)

    utility_fnc_mth = decision_definition['utility_function'][0]
    utility_fnc_params = decision_definition['utility_function'][1]
    utility_fnc = utility_fnc_mth(utility_fnc_params)

    decision_thresholds = decision_definition['decision_thresholds']
    econ_model, fast_spend = decision_definition['economic_model']

    # what decision making method shall we use
    if 'decision_method' not in decision_definition.keys():
        decision_definition['decision_method'] = 'optimise_over_forecast_distribution'

    if decision_definition['decision_method'] == 'critical_probability_threshold_fixed':
        decision_method = 'critical_probability_threshold_fixed'

        # convert probabilistic forecasts to deterministic
        critical_prob_threshold = decision_definition['critical_probability_threshold']
        fcsts = probabilistic_to_deterministic_forecast(fcsts, critical_prob_threshold)
        refs = probabilistic_to_deterministic_forecast(refs, critical_prob_threshold)

    elif decision_definition['decision_method'] == 'critical_probability_threshold_equals_alpha':
        decision_method = 'critical_probability_threshold_equals_alpha'

    elif decision_definition['decision_method'] == 'critical_probability_threshold_max_value':
        decision_method = 'critical_probability_threshold_max_value'

    elif decision_definition['decision_method'] == 'optimise_over_forecast_distribution':
        decision_method = 'optimise_over_forecast_distribution'

    # Pre-calculate the forecast likelihoods for each threshold class
    fcst_likelihoods = all_likelihoods(obs, fcsts, decision_thresholds)
    refs = generate_event_freq_ref(obs) if refs is None else refs
    ref_likelihoods = all_likelihoods(obs, refs, decision_thresholds)

    # Calculate RUV
    results = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, decision_thresholds, econ_model, fast_spend, damage_fnc, utility_fnc, decision_method, parallel_nodes, verbose)

    return {
        'ruv': results['ruv'],
        'fcst_avg_ex_post': results['fcst_avg_ex_post'],
        'obs_avg_ex_post': results['obs_avg_ex_post'],
        'ref_avg_ex_post': results['ref_avg_ex_post'],
        'fcst_spends': results['fcst_spends'],
        'obs_spends': results['obs_spends'],
        'ref_spends': results['ref_spends'],
        'fcst_likelihoods': results['fcst_likelihoods'],
        'ref_likelihoods': results['ref_likelihoods'],
        'fcst_ex_post': results['fcst_ex_post'],
        'obs_ex_post': results['obs_ex_post'],
        'ref_ex_post': results['ref_ex_post'],
        'decision_definition': decision_definition
    }


def relative_utility_value_bootstrap(obs, fcst_ens, ref, decision_definition, parallel_nodes=4, verbose=False, bootstrap_samples=100, confidence_level=0.95):
    start = time.time()
    complete_results = relative_utility_value(obs, fcst_ens, ref, decision_definition, parallel_nodes, verbose=False)
    if verbose: 
        print('Bootstrap will take approximatly %.0f minutes' % ((time.time() - start) * bootstrap_samples / 60))

    global count
    count = 0
    def booty(indices):
        global count; count += 1
        if verbose and count % 5 == 0:
            print('Bootstrap sample: %d / %d' % (count, bootstrap_samples))
        return relative_utility_value(obs[indices], fcst_ens[indices], ref, decision_definition, parallel_nodes, verbose=False)['ruv']

    indices = [np.arange(len(obs))]
    bootstrap_results = bootstrap(indices, booty, n_resamples=bootstrap_samples, confidence_level=confidence_level, vectorized=False, method='percentile')

    return {'bootstrap': bootstrap_results, 'complete': complete_results }
