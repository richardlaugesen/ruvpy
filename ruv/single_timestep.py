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

# TODO: lots of this could be done in place or without creating new variables

# Calculate RUV for a single economic parameter and single timestep
def single_timestep(t: int, econ_par: float, ob: float, fcst: np.array, ref: np.array, context: DecisionContext) -> dict[str, np.ndarray]:
    # ob = data.obs[t]
    # fcst = data.fcsts[t]
    # ref = data.refs[t]

    ob_threshold = realised_threshold(ob, context.decision_thresholds)
    ob_spend = context.analytical_spend(econ_par, ob_threshold, context.damage_function)
    ob_damage = context.damage_function(ob_threshold)

    if is_deterministic(fcst):
        fcst_threshold = realised_threshold(fcst, context.decision_thresholds)
        fcst_spend = context.analytical_spend(econ_par, fcst_threshold, context.damage_function)
        fcst_expected_damage = context.damage_function(fcst_threshold)
    else:
        fcst_likelihoods = calc_likelihood(fcst, context.decision_thresholds)               # not pre-calculating likelihoods because code becomes
        fcst_spend = find_spend_ensemble(econ_par, fcst, fcst_likelihoods, context)        # difficult to maintain even though it is 30% speedup
        if context.decision_thresholds is not None:
            fcst_expected_damage = np.sum(fcst_likelihoods * context.damage_function(context.decision_thresholds))
        else:
            fcst_expected_damage = np.sum(fcst_likelihoods * context.damage_function(fcst))

    if is_deterministic(ref):
        ref_threshold = realised_threshold(ref, context.decision_thresholds)
        ref_spend = context.analytical_spend(econ_par, ref_threshold, context.damage_function)
        ref_expected_damage = context.damage_function(ref_threshold)
    else:
        ref_likelihoods = calc_likelihood(ref, context.decision_thresholds)
        ref_spend = find_spend_ensemble(econ_par, ref, ref_likelihoods, context)
        if context.decision_thresholds is not None:
            ref_expected_damage = np.sum(ref_likelihoods * context.damage_function(context.decision_thresholds))
        else:
            ref_expected_damage = np.sum(ref_likelihoods * context.damage_function(ref))

    # TODO: calc and return the ex_ante utilities

    return {
        't': t,
        'ob_spends': ob_spend,
        'ob_ex_post': ex_post_utility(econ_par, ob_threshold, ob_spend, context),
        'fcst_spend': fcst_spend,
        'fcst_ex_post': ex_post_utility(econ_par, ob_threshold, fcst_spend, context),
        'ref_spend': ref_spend,
        'ref_ex_post': ex_post_utility(econ_par, ob_threshold, ref_spend, context),
        'fcst_expected_damage': fcst_expected_damage,
        'ref_expected_damage': ref_expected_damage,
        'ob_damage': ob_damage,
    }


def find_spend_ensemble(econ_par: float, ens: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:
    if context.decision_thresholds is None:
        # if continuous decision then all members equally likely so thresholds=ens
        curr_context = DecisionContext(context.econ_pars, context.damage_function, context.utility_function, ens, context.economic_model, context.analytical_spend, context.crit_prob_thres)
    else:
        curr_context = context

    def minimise_this(spend):
        return -ex_ante_utility(econ_par, spend, likelihoods, curr_context)

    return minimize_scalar(minimise_this, method='brent').x


def ex_ante_utility(econ_par: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    return np.sum(likelihoods * context.utility_function(context.economic_model(econ_par, context.decision_thresholds, spend, context.damage_function)))


def ex_post_utility(econ_par: float, occured: float, spend: float, context: DecisionContext) -> float:
    return context.utility_function(context.economic_model(econ_par, occured, spend, context.damage_function))


def calc_likelihood(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])   # continuous decision limit is 1/num_classes

    # TODO: this could probably be done in place somehow
    probs_above = ecdf(ens, thresholds)
    adjustment = np.roll(probs_above, -1)
    adjustment[-1] = 0.0
    probs_between = probs_above - adjustment
    probs_between /= np.sum(probs_between)  # normalise to ensure small probs are handled correctly

    return probs_between


def realised_threshold(value: float, thresholds: np.ndarray) -> float:
    if thresholds is None:
        return value

    if np.isnan(value):
        return np.nan

    vals = np.subtract(value, thresholds)
    return thresholds[np.argmin(vals[vals >= 0.0])]
