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

from ruv.multi_timestep import *
from ruv.data_classes import *


def optimise_over_forecast_distribution(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiAlphaOutput:
    outputs = MultiAlphaOutput()
    for alpha in context.alphas:
        outputs.insert(alpha, multiple_timesteps(alpha, data, context, parallel_nodes, verbose))
    return outputs


def critical_probability_threshold_fixed(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiAlphaOutput:
    curr_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, context.crit_prob_thres)
    curr_refs = probabilistic_to_deterministic_forecast(data.refs, context.crit_prob_thres)
    curr_data = InputData(data.obs, curr_fcsts, curr_refs)
    return optimise_over_forecast_distribution(curr_data, context, parallel_nodes, verbose)


def critical_probability_threshold_max_value(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiAlphaOutput:
    outputs = MultiAlphaOutput()
    for alpha in context.alphas:
        def minimise_this(crit_prob_thres):
            curr_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, crit_prob_thres)
            curr_refs = probabilistic_to_deterministic_forecast(data.refs, crit_prob_thres)
            curr_data = InputData(data.obs, curr_fcsts, curr_refs)
            return -multiple_timesteps(alpha, curr_data, context, parallel_nodes, verbose).ruv
        max_ruv_thres = minimize_scalar(minimise_this, method='bounded', bounds=(0, 1), options={'disp': False, 'xatol': 0.005}).x        
        
        max_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, max_ruv_thres)
        max_refs = probabilistic_to_deterministic_forecast(data.refs, max_ruv_thres)
        max_data = InputData(data.obs, max_fcsts, max_refs)
        outputs.insert(alpha, multiple_timesteps(alpha, max_data, context, parallel_nodes, verbose))
    return outputs


def critical_probability_threshold_equals_alpha(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiAlphaOutput:
    outputs = MultiAlphaOutput()
    for alpha in context.alphas:
        curr_fcst = probabilistic_to_deterministic_forecast(data.fcsts, alpha)
        curr_ref = probabilistic_to_deterministic_forecast(data.refs, alpha)
        curr_data = InputData(data.obs, curr_fcst, curr_ref)
        outputs.insert(alpha, multiple_timesteps(alpha, curr_data, context, parallel_nodes, verbose))    
    return outputs
