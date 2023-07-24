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

from ruv.helpers import *
from ruv.data_classes import *
import numpy as np
from scipy.optimize import minimize_scalar


def calc_likelihood_ensemble(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    
    # limit is just 1/num_classes for large num_classes (continuous)
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])    

    probs_above = ecdf(ens, thresholds)
    adjustment = np.roll(probs_above, -1)
    adjustment[-1] = 0.0
    probs_between = np.subtract(probs_above, adjustment)

    # normalise to ensure small probs are handled correctly
    probs_between /= np.sum(probs_between)

    return probs_between


def realised_threshold(value: float, thresholds: np.ndarray) -> float:
    if thresholds is None:
        return value

    if np.isnan(value):
        return np.nan

    vals = np.subtract(value, thresholds)
    idx = np.argmin(vals[vals >= 0.0])
    return thresholds[idx]


def probabilistic_to_deterministic_forecast(fcsts: np.ndarray, crit_thres: float) -> np.ndarray:
    return np.nanquantile(fcsts, 1 - crit_thres, axis=1)


def ex_ante_utility(alpha: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    net_expenses = context.economic_model(alpha, context.decision_thresholds, spend, context.damage_function)
    expected_utility = np.sum(np.multiply(likelihoods, context.utility_function(net_expenses)))
    return expected_utility


def ex_post_utility(alpha: float, occured: float, spend: float, context: DecisionContext) -> float:
    net_expense = context.economic_model(alpha, occured, spend, context.damage_function)
    return context.utility_function(net_expense)


def find_spend(alpha: float, fcst: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:
    # continuous? then all members equally likely
    if context.decision_thresholds is None:
        curr_context = DecisionContext(context.alphas, context.damage_function, context.utility_function, fcst, context.economic_model, context.analytical_spend, context.crit_prob_thres)
    else:
        curr_context = context

    def minimise_this(spend):
        return -ex_ante_utility(alpha, spend, likelihoods, curr_context)
    spend_amount = minimize_scalar(minimise_this, method='brent').x
    return spend_amount


def single_timestep(t: int, alpha: float, data: InputData, context: DecisionContext) -> tuple:
    ob = data.obs[t]
    fcst = data.fcsts[t]
    ref = data.refs[t]

    # find obs spend amount
    ob_threshold = realised_threshold(ob, context.decision_thresholds)
    obs_spends = context.analytical_spend(alpha, ob_threshold, context.damage_function)

    # find fcst and ref spend amounts
    if is_deterministic_timestep(fcst):
        fcst_spends = context.analytical_spend(alpha, realised_threshold(fcst, context.decision_thresholds), context.damage_function)
        ref_spends = context.analytical_spend(alpha, realised_threshold(ref, context.decision_thresholds), context.damage_function)      # TODO: will fail for when refs=None because it needs to be ensemble for some reason, to calc the average I suppose
        
    else:
        fcst_likelihoods = calc_likelihood_ensemble(fcst, context.decision_thresholds)   # not pre-calculating these because code difficult to
        ref_likelihoods = calc_likelihood_ensemble(ref, context.decision_thresholds)     # maintain even though it is 30% faster

        fcst_spends = find_spend(alpha, fcst, fcst_likelihoods, context)
        ref_spends = find_spend(alpha, ref, ref_likelihoods, context)

    # calculate ex post utilities using the spend amounts
    obs_ex_post = ex_post_utility(alpha, ob_threshold, obs_spends, context)
    fcst_ex_post = ex_post_utility(alpha, ob_threshold, fcst_spends, context)
    ref_ex_post = ex_post_utility(alpha, ob_threshold, ref_spends, context)

    return (t, obs_spends, obs_ex_post, fcst_spends, fcst_ex_post, ref_spends, ref_ex_post)

