# Copyright 2024 Richard Laugesen

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

from ruvpy.helpers import generate_event_freq_ref
from ruvpy.data_classes import DecisionContext


# main entry function for RUV calculation
def relative_utility_value(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, decision_definition: dict, parallel_nodes: int=4) -> dict:

    # decision type
    decision_thresholds = decision_definition['decision_thresholds']

    # decision-making method
    decision_rule = decision_definition['decision_rule'][0]
    decision_rule_params = decision_definition['decision_rule'][1]
    decision_rule_fnc = decision_rule(decision_rule_params)

    # damage function
    damage_fnc_mth = decision_definition['damage_function'][0]
    damage_fnc_params = decision_definition['damage_function'][1]
    damage_function = damage_fnc_mth(damage_fnc_params)

    # utility function
    utility_fnc_mth = decision_definition['utility_function'][0]
    utility_fnc_params = decision_definition['utility_function'][1]
    utility_function = utility_fnc_mth(utility_fnc_params)

    # economic model
    economic_model_fnc = decision_definition['economic_model'][0]
    economic_model_analytical_spend_fnc = decision_definition['economic_model'][1]
    economic_model_params = decision_definition['economic_model'][2]

    context_fields = {
        'economic_model_params': economic_model_params,
        'damage_function': damage_function,
        'utility_function': utility_function,
        'decision_thresholds': decision_thresholds,
        'economic_model': economic_model_fnc,
        'analytical_spend': economic_model_analytical_spend_fnc,
        'decision_rule': decision_rule_fnc
    }
    context = DecisionContext(**context_fields)

    # generate event frequency refs if requested
    if refs is None:
        refs = generate_event_freq_ref(obs)

    _check_inputs(obs, fcsts, refs, context)
    results = context.decision_rule(obs, fcsts, refs, context, parallel_nodes)

    return results.to_dict()


def _check_inputs(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext) -> None:

    context.validate_fields()

    if np.any(np.isnan(fcsts)) or (refs is not None and np.any(np.isnan(refs))):
        raise ValueError('Cannot calculate RUV with missing values in forecasts or references')

    if context.decision_thresholds is not None:
        if np.any(obs < np.min(context.decision_thresholds)) or np.any(fcsts < np.min(context.decision_thresholds)) or (refs is not None and np.any(refs < np.min(context.decision_thresholds))):
            raise ValueError('One or more values are less than smallest threshold')

    if len(obs) != len(fcsts) or (refs is not None and len(obs) != len(refs)):
        raise ValueError('Lengths of obs, fcsts and refs must be the same')

