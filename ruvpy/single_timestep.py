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
from scipy.optimize import minimize_scalar

from ruvpy.helpers import is_deterministic, ecdf
from ruvpy.data_classes import DecisionContext


# Calculate RUV for a single economic parameter and single timestep
def single_timestep(t: int, econ_par: float, ob: float, fcst: np.array, ref: np.array, context: DecisionContext) -> dict[str, np.ndarray]:
    ob_threshold = _realised_threshold(ob, context.decision_thresholds)
    ob_spend = context.analytical_spend(econ_par, ob_threshold, context.damage_function)

    if is_deterministic(fcst):
        fcst_threshold = _realised_threshold(fcst, context.decision_thresholds)
        fcst_spend = context.analytical_spend(econ_par, fcst_threshold, context.damage_function)
    else:
        fcst_likelihoods = _calc_likelihood(fcst, context.decision_thresholds)              # not pre-calculating likelihoods because code becomes
        fcst_spend = _find_spend_ensemble(econ_par, fcst, fcst_likelihoods, context)        # difficult to read and maintain even though it is
                                                                                            # an approximately 30% speedup
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

    return minimize_scalar(minimise_this, method='brent').x


def _ex_ante_utility(econ_par: float, spend: float, likelihoods: np.ndarray, context: DecisionContext) -> float:
    net_outcome = context.economic_model(econ_par, context.decision_thresholds, spend, context.damage_function)
    return np.dot(likelihoods, context.utility_function(net_outcome))


def _ex_post_utility(econ_par: float, occurred: float, spend: float, context: DecisionContext) -> float:
    net_outcome = context.economic_model(econ_par, occurred, spend, context.damage_function)
    return context.utility_function(net_outcome)


def _calc_likelihood(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if thresholds is None:
        return np.full(ens.shape, 1/ens.shape[0])   # continuous decision limit is 1/num_classes

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
