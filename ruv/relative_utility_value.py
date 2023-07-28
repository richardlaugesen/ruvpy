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

import ruv.decision_methods
from ruv.data_classes import *
import numpy as np


# main entry function for RUV calculation
def relative_utility_value(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, decision_definition: dict, parallel_nodes: int=4, verbose: bool=False) -> dict:
    data = InputData(obs, fcsts, refs)

    alphas = decision_definition['alphas']
    damage_fnc_mth = decision_definition['damage_function'][0]
    damage_fnc_params = decision_definition['damage_function'][1]
    damage_function = damage_fnc_mth(damage_fnc_params)
    utility_fnc_mth = decision_definition['utility_function'][0]
    utility_fnc_params = decision_definition['utility_function'][1]
    utility_function = utility_fnc_mth(utility_fnc_params)
    decision_thresholds = decision_definition['decision_thresholds']
    economic_model, analytical_spend = decision_definition['economic_model']
    crit_prob_thres = decision_definition['critical_probability_threshold'] if 'critical_probability_threshold' in decision_definition.keys() else None
    decision_method = decision_definition['decision_method'] if 'decision_method' in decision_definition.keys() else 'optimise_over_forecast_distribution'
    decision_making_fnc = getattr(ruv.decision_methods, decision_method)
    event_freq_ref = decision_definition['event_freq_ref'] if 'event_freq_ref' in decision_definition.keys() else False
    context = DecisionContext(alphas, damage_function, utility_function, decision_thresholds, economic_model, analytical_spend, crit_prob_thres, event_freq_ref)

    check_inputs(data, context)
    results = decision_making_fnc(data, context, parallel_nodes, verbose)    
    return to_dict(results)


def check_inputs(data: InputData, context: DecisionContext) -> None:  
    if np.any(np.isnan(data.fcsts)) or (data.refs is not None and np.any(np.isnan(data.refs))):
        raise ValueError('Cannot calculate RUV with missing values in forecasts or references')

    if context.decision_thresholds is not None:
        if np.any(data.obs < np.min(context.decision_thresholds)) or np.any(data.fcsts < np.min(context.decision_thresholds)) or (data.refs is not None and np.any(data.refs < np.min(context.decision_thresholds))):
            raise ValueError('One or more values are less than smallest threshold')


# TODO: return ex ante utility too
def to_dict(outputs: MultiAlphaOutput) -> dict:
    results = {}
    results['ruv'] = outputs.get_series('ruv')
    results['avg_fcst_ex_post'] = outputs.get_series('avg_fcst_ex_post')
    results['avg_obs_ex_post'] = outputs.get_series('avg_obs_ex_post')
    results['avg_ref_ex_post'] = outputs.get_series('avg_ref_ex_post')
    results['fcst_spends'] = outputs.get_series('fcst_spends')
    results['obs_spends'] = outputs.get_series('obs_spends')
    results['ref_spends'] = outputs.get_series('ref_spends')
    results['fcst_ex_post'] = outputs.get_series('fcst_ex_post')
    results['obs_ex_post'] = outputs.get_series('obs_ex_post')
    results['ref_ex_post'] = outputs.get_series('ref_ex_post')
    results['fcst_likelihoods'] = outputs.get_series('fcst_likelihoods')
    results['ref_likelihoods'] = outputs.get_series('ref_likelihoods')
    return results