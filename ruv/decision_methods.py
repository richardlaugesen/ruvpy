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
from ruv.helpers import *

# generate event frequency refs if requested, otherwise leave refs as supplied 
# so impact on value from decision making method depends on forecasts alone

# TODO: think more carefully whether we want to convert ref to deterministic in all the threshold based methods

def optimise_over_forecast_distribution(data: InputData, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)
    updated_data = InputData(data.obs, data.fcsts, refs)

    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        outputs.insert(econ_par, multiple_timesteps(econ_par, updated_data, context, parallel_nodes))
    return outputs


def critical_probability_threshold_fixed(data: InputData, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)

    fcsts = probabilistic_to_deterministic_forecast(data.fcsts, context.crit_prob_thres)
    #refs = probabilistic_to_deterministic_forecast(refs, context.crit_prob_thres)     # TODO: this makes value diagrams flat
    updated_data = InputData(data.obs, fcsts, refs)

    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        outputs.insert(econ_par, multiple_timesteps(econ_par, updated_data, context, parallel_nodes))
    return outputs


def critical_probability_threshold_max_value(data: InputData, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)
    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        def minimise_this(crit_prob_thres):
            curr_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, crit_prob_thres)
            curr_refs = refs #probabilistic_to_deterministic_forecast(refs, crit_prob_thres)    # TODO: this could maximise RUV by adjusting threshold to make the ref worse rather than fcst better
            curr_data = InputData(data.obs, curr_fcsts, curr_refs)
            return -multiple_timesteps(econ_par, curr_data, context, parallel_nodes).ruv
        max_ruv_thres = minimize_scalar(minimise_this, method='bounded', bounds=(0, 1), options={'disp': False, 'xatol': 0.005}).x        

        max_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, max_ruv_thres)
        #max_refs = probabilistic_to_deterministic_forecast(refs, max_ruv_thres)
        max_data = InputData(data.obs, max_fcsts, refs)
        outputs.insert(econ_par, multiple_timesteps(econ_par, max_data, context, parallel_nodes))

    return outputs


def critical_probability_threshold_equals_par(data: InputData, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)
    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        curr_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, econ_par)
        #curr_refs = probabilistic_to_deterministic_forecast(refs, econ_par) # TODO: this one is possibly okay
        curr_data = InputData(data.obs, curr_fcsts, refs)
        outputs.insert(econ_par, multiple_timesteps(econ_par, curr_data, context, parallel_nodes))
    
    return outputs

