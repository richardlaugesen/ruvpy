
# Copyright 2024 RUVPY Developers

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable
import copy
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar, brute

from ruvpy.helpers import is_deterministic, ecdf, nanmode
from ruvpy.data_classes import DecisionContext


# Calculate RUV for a single economic parameter and single timestep
def single_timestep(t: int, econ_par: float, ob: float, fcst: np.array, ref: np.array, context: DecisionContext) -> dict[str, np.ndarray]:
    #print(f"-------- NEW TIMESTEP {t} / {econ_par} -----------")
    
    #print('Calculating observation')
    ob_threshold = _realised_threshold(ob, context.decision_thresholds)
    ob_spend = context.analytical_spend(econ_par, ob_threshold, context.damage_function)

    #print('Optimising forecast')
    if is_deterministic(fcst):
        fcst_threshold = _realised_threshold(fcst, context.decision_thresholds)
        fcst_spend = context.analytical_spend(econ_par, fcst_threshold, context.damage_function)
    else:
        fcst_likelihoods = _calc_likelihood(fcst, context.decision_thresholds)
        fcst_spend = _find_spend_ensemble(econ_par, fcst, fcst_likelihoods, context)

        # not pre-calculating likelihoods because code becomes difficult to read and maintain even
        # though it is an approximately 30% speedup

    #print('Optimising reference')        
    if is_deterministic(ref):
        ref_threshold = _realised_threshold(ref, context.decision_thresholds)
        ref_spend = context.analytical_spend(econ_par, ref_threshold, context.damage_function)
    else:
        ref_likelihoods = _calc_likelihood(ref, context.decision_thresholds)
        ref_spend = _find_spend_ensemble(econ_par, ref, ref_likelihoods, context)

    #avg_net_outcome = np.mean(context.economic_model(econ_par, context.decision_thresholds, fcst_spend, context.damage_function))
    #print('timestep: %d, ob: %.1f ob_spend: $%.6f, fcst_spend: $%.6f ($%.6f), ref_spend: $%.6f' % (t, ob, ob_spend, fcst_spend, avg_net_outcome, ref_spend))

    return {
        't': t,
        'ob_spend': ob_spend,
        'ob_ex_post': _ex_post_utility(econ_par, ob_threshold, ob_spend, context),
        'fcst_spend': fcst_spend,
        'fcst_ex_post': _ex_post_utility(econ_par, ob_threshold, fcst_spend, context),
        'ref_spend': ref_spend,
        'ref_ex_post': _ex_post_utility(econ_par, ob_threshold, ref_spend, context)
    }


def _find_spend_ensemble_debug(econ_par: float, ens: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:
    # Set up decision thresholds if they are None
    if context.decision_thresholds is None:
        context = copy.deepcopy(context)
        context.decision_thresholds = ens

    # Define the function to minimise (negative of utility)
    def minimise_this(spend):
        return -_ex_ante_utility(econ_par, spend, likelihoods, context)

    # Define bounds for spend
    lower_bound = 0
    upper_bound = 2 * context.max_damages
    bounds = [(lower_bound, upper_bound)]

    # Brute-force search
    #print('brute force')
    step_size = context.max_damages / 1e4
    ranges = (slice(lower_bound, upper_bound, step_size),)
    brute_result = brute(minimise_this, ranges, full_output=True, finish=None)
    brute_opt_spend = brute_result[0]
    brute_opt_value = brute_result[1]

    # Differential Evolution: Multiple Runs
    #print('differential evolution')
    de_results = []
    for seed in range(10):
        result_de = differential_evolution(
            minimise_this,
            bounds,
            strategy='best1bin',
            mutation=(0.7, 1.0),
            recombination=0.9,
            maxiter=200,
            popsize=5,
            tol=1e-3,
            seed=seed
        )
        de_results.append((result_de.x[0], result_de.fun))

    de_best_result = min(de_results, key=lambda x: x[1])    
    de_best_opt_spend = de_best_result[0]
    de_best_opt_value = de_best_result[1]
        
    # Minimize Scalar (Brent method)
    #print('minimize scalar')

    result_brent = minimize_scalar(
        minimise_this,
        bracket=(lower_bound, upper_bound),
        method='brent'
    )
    brent_opt_spend = result_brent.x
    brent_opt_value = result_brent.fun

    # Compare and return the best result    
    all_results = [
        ("Brute-force", brute_opt_spend, -brute_opt_value),
        ("DE Best", de_best_opt_spend, -de_best_opt_value),
        ("Minimize_scalar", brent_opt_spend, -brent_opt_value)
    ]

    brute_force_spend = all_results[0][1]

    rtol = 0.1
    de_close_brute = np.isclose(all_results[0][1], all_results[1][1], rtol=rtol)
    de_close_min_scal = np.isclose(all_results[1][1], all_results[2][1], rtol=rtol)
    min_scal_close_brute = np.isclose(all_results[0][1], all_results[2][1], rtol=rtol)

    from matplotlib import pyplot as plt
    if (all_results[0][1] > 1) or ((not de_close_min_scal) and (all_results[0][1] + all_results[1][1] > 0.01)):

        plt.hist(ens)
        for threshold in context.decision_thresholds:    
            plt.axvline(threshold, color='grey', label='thres=%.1f' % threshold)
        plt.title('Forecast ensemble')
        plt.ylabel('Forecast likelihood')
        plt.xlabel('Streamflow (ML/d)')
        #plt.legend()
        plt.show()
        
        spends = np.arange(lower_bound, upper_bound, step_size)
        utilities = [-minimise_this(spend) for spend in spends]
        plt.plot(spends, utilities, label='surface', color='blue')
        plt.axvline(all_results[0][1], label='brute force', color='red')
        plt.axvline(all_results[1][1], label='differential evolution', color='green')
        plt.axvline(all_results[2][1], label='scalar minimisation', color='purple')
        plt.title('Optimisation surface')
        plt.xlabel('Spend ($)')
        plt.ylabel('Utility')
        plt.legend()
        plt.show()

        if not de_close_min_scal:
            print('differential_evolution is more than %.2f relative tolerance from minimize_scalar' % rtol)
       
        print('\nRange [%.3f, %.3f], brute_force step_size of %.3f' %(lower_bound, upper_bound, step_size))

        for method, spend, utility in all_results:
            print(f"{method}: Spend = {spend:.3f}, Utility = {utility:.3f}")

        print('\nDifferential evolution runs:')
        for i, (spend, utility) in enumerate(de_results):
            print(f"DE Run {i+1}: Spend = {spend:.3f}, Utility = {-utility:.3f}")
        
        #import pdb; pdb.set_trace()

    if de_spend > (0.001 * context.max_damages):
        def bench_this(tol, pop, mut, rec, polish):
            de_result = differential_evolution(minimise_this,
                                               [(lower_bound, upper_bound)],
                                               strategy='best1bin',
                                               #recombination=rec,
                                               tol=tol,
                                               popsize=pop,
                                               polish=polish,
                                               #seed=3,
                                               #mutation=mut
                                               )
            de_spend = de_result.x[0]
            print(f"TOL={tol:0.7f} POP={pop} REC={rec:.2f} MUT={mut} POLISH={polish} OPT_SPEND={de_spend:.7f}")
            return de_spend

        import timeit
        spends = {}
        for polish in [True, False]:
            for mut in [0]: #np.arange(0.4, 1.8, 0.2):
                for rec in [0]: #np.arange(0, 1, 0.2):
                    for pop in [10]: #[30, 15, 10, 5, 1]:
                        for tol in [1E-6, 1E-5, 1E-4, 1E-3, 1E-2]:
                            print(f"------ POP={pop} / TOL={tol} / MUT={mut:.2f} / REC={rec:.2f} / POLISH={polish} ------")
                            times = timeit.timeit(lambda: bench_this(tol, pop, mut, rec, polish), number=10)
                            spends[f"{pop}_{tol}_{mut:.2f}_{rec:.2f}_{polish}"] = times
                            print(f"Average time: {times:.2f} seconds")
                            
        for key, val in spends.items():
            print(f"{key} took: {val: .2f} seconds")

        def_time = timeit.timeit(lambda: differential_evolution(minimise_this, [(lower_bound, upper_bound)]),number=10)
        print(f"Default time: {def_time:.2f} seconds")
                
        import pdb; pdb.set_trace()
               
    if brute_force_spend > step_size: # avoids false positives
        if not min_scal_close_brute:
            print('minimize_scalar is more than %.2f relative tolerance from brute_force' % rtol)

        if not de_close_brute:            
            print('differential_evolution is more than %.2f relative tolerance from brute force' % rtol)
            
    return brute_force_spend


def _find_spend_ensemble(econ_par: float, ens: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:

    # if continuous decision then all members are equally likely so thresholds=ens
    if context.decision_thresholds is None:
        context = copy.deepcopy(context)
        context.decision_thresholds = ens

    def minimise_this(spend):
        return -_ex_ante_utility(econ_par, spend, likelihoods, context)

    # TODO: these specific to cost-loss but need to be general
    lower_bound = 0
    upper_bound = 2 * context.max_damages
    bounds = [(lower_bound, upper_bound)]

    result = differential_evolution(minimise_this, [(lower_bound, upper_bound)], polish=context.polish)
    spend = result.x[0]

    if not result.success:
        print(f'\033[1;31mDifferential evolution failed: {result.message}\033[0m')

    return spend


def _ex_ante_utility(econ_par: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    net_outcome = context.economic_model(econ_par, context.decision_thresholds, spend, context.damage_function)

    # TODO: use some proper way to choose normative or descriptive
    if context.reference_point is None:
        # Normative decision-maker with expected utility theory
        utilities = context.utility_function(net_outcome)
        return np.dot(likelihoods, utilities)

    else:
        # Descriptive decision-maker with cumulative prospect theory
        deviations = net_outcome - context.reference_point
        prospects = context.utility_function(deviations)

        # Calculate prospect value of gains
        weighted_gain_prospect = 0
        gains = deviations >= 0
        if np.any(gains):
            gain_prospects = prospects[gains]
            gain_likelihoods = likelihoods[gains]
            gain_indices = np.argsort(gain_prospects)
            sorted_gain_prospects = gain_prospects[gain_indices]
            sorted_gain_likelihoods = gain_likelihoods[gain_indices]
            cumulative_gain_likelihoods = np.cumsum(sorted_gain_likelihoods)
            weighted_gain_likelihoods = context.probability_weight_function(cumulative_gain_likelihoods)
            gain_decision_weights = np.diff(np.insert(weighted_gain_likelihoods, 0, 0))
            weighted_gain_prospect = np.dot(gain_decision_weights, sorted_gain_prospects)

        # Calculate prospect value of losses
        weighted_loss_prospect = 0
        losses = ~gains
        if np.any(losses):
            loss_prospects = prospects[losses]
            loss_likelihoods = likelihoods[losses]
            loss_indices = np.argsort(-loss_prospects)  # sort losses in descending order
            sorted_loss_prospects = loss_prospects[loss_indices]
            sorted_loss_likelihoods = loss_likelihoods[loss_indices]
            cumulative_loss_likelihoods = np.cumsum(sorted_loss_likelihoods)
            weighted_loss_likelihoods = context.probability_weight_function(cumulative_loss_likelihoods)
            loss_decision_weights = np.diff(np.insert(weighted_loss_likelihoods, 0, 0))
            weighted_loss_prospect = np.dot(loss_decision_weights, sorted_loss_prospects)

        return weighted_gain_prospect + weighted_loss_prospect


def _ex_post_utility(econ_par: float, occurred: float, spend: float, context: DecisionContext) -> float:
    net_outcome = context.economic_model(econ_par, occurred, spend, context.damage_function)

    # TODO: use some proper way to choose normative or descriptive    
    if context.reference_point is None:
        # Normative decision-maker with expected utility theory
        return context.utility_function(net_outcome)
    else:
        # Descriptive decision-maker with cumulative prospect theory
        return context.utility_function(net_outcome - context.reference_point)


def _calc_likelihood(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])   # continuous decision limit is 1/num_classes (equally likely)

    probs_above = ecdf(ens, thresholds)
    adjustment = np.roll(probs_above, -1)
    adjustment[-1] = 0.0
    likelihoods = np.subtract(probs_above, adjustment)
    likelihoods = np.divide(likelihoods, np.sum(likelihoods))  # normalise to ensure small probs are handled correctly

    return likelihoods


def _realised_threshold(value: float, thresholds: np.ndarray) -> float:
    if thresholds is None:
        return value

    if np.isnan(value):
        return np.nan

    vals = np.subtract(value, thresholds)
    return thresholds[np.argmin(vals[vals >= 0.0])]


