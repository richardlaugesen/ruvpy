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


# Calculate RUV for a single alpha and single timestep
def single_timestep(t: int, alpha: float, data: InputData, context: DecisionContext) -> tuple:
    ob = data.obs[t]
    fcst = data.fcsts[t]
    ref = data.refs[t]

    ob_threshold = realised_threshold(ob, context.decision_thresholds)
    obs_spends = context.analytical_spend(alpha, ob_threshold, context.damage_function)

    if is_deterministic(fcst):
        fcst_spends = context.analytical_spend(alpha, realised_threshold(fcst, context.decision_thresholds), context.damage_function)
    else:
        fcst_likelihoods = calc_likelihood(fcst, context.decision_thresholds)   # not pre-calculating these because code difficult to
        fcst_spends = find_spend(alpha, fcst, fcst_likelihoods, context)        # maintain even though it is 30% faster
    
    if is_deterministic(ref):
        ref_spends = context.analytical_spend(alpha, realised_threshold(ref, context.decision_thresholds), context.damage_function)        
    else:
        ref_likelihoods = calc_likelihood(ref, context.decision_thresholds)
        ref_spends = find_spend(alpha, ref, ref_likelihoods, context)     

    obs_ex_post = ex_post_utility(alpha, ob_threshold, obs_spends, context)
    fcst_ex_post = ex_post_utility(alpha, ob_threshold, fcst_spends, context)
    ref_ex_post = ex_post_utility(alpha, ob_threshold, ref_spends, context)

    return (t, obs_spends, obs_ex_post, fcst_spends, fcst_ex_post, ref_spends, ref_ex_post)


def find_spend(alpha: float, fcst: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:    
    if context.decision_thresholds is None:
        thresholds = fcst   # if continuous decision then all members equally likely
        curr_context = DecisionContext(context.alphas, context.damage_function, context.utility_function, thresholds, context.economic_model, context.analytical_spend, context.crit_prob_thres)
    else:
        curr_context = context

    def minimise_this(spend):
        return -ex_ante_utility(alpha, spend, likelihoods, curr_context)
    spend_amount = minimize_scalar(minimise_this, method='brent').x

    return spend_amount


def ex_ante_utility(alpha: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    net_expenses = context.economic_model(alpha, context.decision_thresholds, spend, context.damage_function)
    expected_utility = np.sum(np.multiply(likelihoods, context.utility_function(net_expenses)))
    return expected_utility


def ex_post_utility(alpha: float, occured: float, spend: float, context: DecisionContext) -> float:
    net_expense = context.economic_model(alpha, occured, spend, context.damage_function)
    return context.utility_function(net_expense)


def calc_likelihood(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:    
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])   # continuous decision limit is 1/num_classes

    probs_above = ecdf(ens, thresholds)
    adjustment = np.roll(probs_above, -1)
    adjustment[-1] = 0.0
    probs_between = np.subtract(probs_above, adjustment)
    probs_between /= np.sum(probs_between)  # normalise to ensure small probs are handled correctly

    return probs_between


def realised_threshold(value: float, thresholds: np.ndarray) -> float:
    if thresholds is None:
        return value

    if np.isnan(value):
        return np.nan

    vals = np.subtract(value, thresholds)
    idx = np.argmin(vals[vals >= 0.0])
    return thresholds[idx]
