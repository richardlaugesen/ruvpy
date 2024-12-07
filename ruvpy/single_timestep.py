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

import numpy as np
import copy
from scipy.optimize import differential_evolution, minimize_scalar, brute

from ruvpy.helpers import is_deterministic, ecdf, nanmode
from ruvpy.data_classes import DecisionContext


# Calculate RUV for a single economic parameter and single timestep
def single_timestep(t: int, econ_par: float, ob: float, fcst: np.array, ref: np.array, context: DecisionContext) -> dict[str, np.ndarray]:
    ob_threshold = _realised_threshold(ob, context.decision_thresholds)
    ob_spend = context.analytical_spend(econ_par, ob_threshold, context.damage_function)

    if is_deterministic(fcst):
        fcst_threshold = _realised_threshold(fcst, context.decision_thresholds)
        fcst_spend = context.analytical_spend(econ_par, fcst_threshold, context.damage_function)
    else:
        fcst_likelihoods = _calc_likelihood(fcst, context.decision_thresholds)
        fcst_spend = _find_spend_ensemble(econ_par, fcst, fcst_likelihoods, context)

        # not pre-calculating likelihoods because code becomes difficult to read and maintain even
        # though it is an approximately 30% speedup

    if is_deterministic(ref):
        ref_threshold = _realised_threshold(ref, context.decision_thresholds)
        ref_spend = context.analytical_spend(econ_par, ref_threshold, context.damage_function)
    else:
        ref_likelihoods = _calc_likelihood(ref, context.decision_thresholds)
        ref_spend = _find_spend_ensemble(econ_par, ref, ref_likelihoods, context)

    return {
        't': t,
        'ob_spend': ob_spend,
        'ob_ex_post': _ex_post_utility(econ_par, ob_threshold, ob_spend, context),
        'fcst_spend': fcst_spend,
        'fcst_ex_post': _ex_post_utility(econ_par, ob_threshold, fcst_spend, context),
        'ref_spend': ref_spend,
        'ref_ex_post': _ex_post_utility(econ_par, ob_threshold, ref_spend, context)
    }


def _find_spend_ensemble(econ_par: float, ens: np.ndarray, likelihoods: np.ndarray, context: DecisionContext) -> float:

    # if continuous decision then all members are equally likely so thresholds=ens
    if context.decision_thresholds is None:
        context = copy.deepcopy(context)
        context.decision_thresholds = ens

    def minimise_this(spend):
        return -_ex_ante_utility(econ_par, spend, likelihoods, context)

    lower_bound = context.optimiser['lower_bound']
    upper_bound = context.optimiser['upper_bound']

    bounds = [(lower_bound, upper_bound)]
    result = differential_evolution(minimise_this,
                                    [(lower_bound, upper_bound)],
                                    tol=context.optimiser['tolerance'],
                                    seed = context.optimiser['seed'],
                                    polish=context.optimiser['polish'])
    spend = result.x[0]

    if not result.success:
        print(f'\033[1;31mOptimisation failed: {result.message}\033[0m')

    return spend


def _ex_ante_utility(econ_par: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    net_outcome = context.economic_model(econ_par, context.decision_thresholds, spend, context.damage_function)

    # TODO: use some proper way to detect normative or descriptive
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
