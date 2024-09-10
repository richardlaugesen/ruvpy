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

from ruv.helpers import is_deterministic, ecdf
from ruv.data_classes import DecisionContext

# TODO: lots of this could be done in place or without creating new variables

# Calculate RUV for a single economic parameter and single timestep
def single_timestep(t: int, econ_par: float, ob: float, fcst: np.array, ref: np.array, context: DecisionContext) -> dict[str, np.ndarray]:
    ob_threshold = _realised_threshold(ob, context.decision_thresholds)
    ob_spend = context.analytical_spend(econ_par, ob_threshold, context.damage_function)
    ob_damage = context.damage_function(ob_threshold)

    if is_deterministic(fcst):
        fcst_threshold = _realised_threshold(fcst, context.decision_thresholds)
        fcst_spend = context.analytical_spend(econ_par, fcst_threshold, context.damage_function)
        fcst_expected_damage = context.damage_function(fcst_threshold)
    else:
        fcst_likelihoods = _calc_likelihood(fcst, context.decision_thresholds)               # not pre-calculating likelihoods because code becomes
        fcst_spend = _find_spend_ensemble(econ_par, fcst, fcst_likelihoods, context)        # difficult to maintain even though it is 30% speedup
        if context.decision_thresholds is not None:
            fcst_expected_damage = np.dot(fcst_likelihoods, context.damage_function(context.decision_thresholds))
        else:
            fcst_expected_damage = np.dot(fcst_likelihoods, context.damage_function(fcst))

    if is_deterministic(ref):
        ref_threshold = _realised_threshold(ref, context.decision_thresholds)
        ref_spend = context.analytical_spend(econ_par, ref_threshold, context.damage_function)
        ref_expected_damage = context.damage_function(ref_threshold)
    else:
        ref_likelihoods = _calc_likelihood(ref, context.decision_thresholds)
        ref_spend = _find_spend_ensemble(econ_par, ref, ref_likelihoods, context)
        if context.decision_thresholds is not None:
            ref_expected_damage = np.dot(ref_likelihoods, context.damage_function(context.decision_thresholds))
        else:
            ref_expected_damage = np.dot(ref_likelihoods, context.damage_function(ref))

    # TODO: calc and return the ex_ante utilities
    return {
        't': t,
        'ob_spend': ob_spend,
        'ob_ex_post': _ex_post_utility(econ_par, ob_threshold, ob_spend, context),
        'fcst_spend': fcst_spend,
        'fcst_ex_post': _ex_post_utility(econ_par, ob_threshold, fcst_spend, context),
        'ref_spend': ref_spend,
        'ref_ex_post': _ex_post_utility(econ_par, ob_threshold, ref_spend, context),
        'fcst_expected_damage': fcst_expected_damage,
        'ref_expected_damage': ref_expected_damage,
        'ob_damage': ob_damage,
    }


def _find_spend_ensemble(econ_par: float, ens: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:
    if context.decision_thresholds is None:
        # if continuous decision then all members equally likely so thresholds=ens
        context_fields = {
            'economic_model_params': context.economic_model_params,
            'damage_function': context.damage_function,
            'utility_function': context.utility_function,
            'economic_model': context.economic_model,
            'analytical_spend': context.analytical_spend,
            'decision_making_method': context.decision_making_method,
            'decision_thresholds': ens
        }
        curr_context = DecisionContext(**context_fields)
    else:
        curr_context = context

    def minimise_this(spend):
        return -_ex_ante_utility(econ_par, spend, likelihoods, curr_context)

    return minimize_scalar(minimise_this, method='brent').x


def _ex_ante_utility(econ_par: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    return np.dot(likelihoods, context.utility_function(context.economic_model(econ_par, context.decision_thresholds, spend, context.damage_function)))


def _ex_post_utility(econ_par: float, occured: float, spend: float, context: DecisionContext) -> float:
    return context.utility_function(context.economic_model(econ_par, occured, spend, context.damage_function))


def _calc_likelihood(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])   # continuous decision limit is 1/num_classes

    # TODO: this could probably be done in place somehow
    probs_above = ecdf(ens, thresholds)
    adjustment = np.roll(probs_above, -1)
    adjustment[-1] = 0.0
    probs_between = np.subtract(probs_above, adjustment)
    probs_between = np.divide(probs_between, np.sum(probs_between))  # normalise to ensure small probs are handled correctly

    return probs_between


def _realised_threshold(value: float, thresholds: np.ndarray) -> float:
    if thresholds is None:
        return value

    if np.isnan(value):
        return np.nan

    vals = np.subtract(value, thresholds)
    return thresholds[np.argmin(vals[vals >= 0.0])]
