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

# generate event frequency refs if requested, otherwise leave refs as supplied 
# so impact on value from decision making method depends on forecasts alone

# TODO: think more carefully whether we want to convert ref to deterministic in all the threshold based methods

def optimise_over_forecast_distribution(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)
    updated_data = InputData(data.obs, data.fcsts, refs)

    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        outputs.insert(econ_par, multiple_timesteps(econ_par, updated_data, context, parallel_nodes, verbose))
    return outputs


def critical_probability_threshold_fixed(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)

    fcsts = probabilistic_to_deterministic_forecast(data.fcsts, context.crit_prob_thres)
    #refs = probabilistic_to_deterministic_forecast(refs, context.crit_prob_thres)     # TODO: this makes value diagrams flat
    updated_data = InputData(data.obs, fcsts, refs)

    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        outputs.insert(econ_par, multiple_timesteps(econ_par, updated_data, context, parallel_nodes, verbose))
    return outputs


def critical_probability_threshold_max_value(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)
    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        def minimise_this(crit_prob_thres):
            curr_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, crit_prob_thres)
            curr_refs = refs #probabilistic_to_deterministic_forecast(refs, crit_prob_thres)    # TODO: this could maximise RUV by adjusting threshold to make the ref worse rather than fcst better
            curr_data = InputData(data.obs, curr_fcsts, curr_refs)
            return -multiple_timesteps(econ_par, curr_data, context, parallel_nodes, verbose).ruv
        max_ruv_thres = minimize_scalar(minimise_this, method='bounded', bounds=(0, 1), options={'disp': False, 'xatol': 0.005}).x        

        max_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, max_ruv_thres)
        #max_refs = probabilistic_to_deterministic_forecast(refs, max_ruv_thres)
        max_data = InputData(data.obs, max_fcsts, refs)
        outputs.insert(econ_par, multiple_timesteps(econ_par, max_data, context, parallel_nodes, verbose))

    return outputs


def critical_probability_threshold_equals_par(data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> MultiParOutput:
    refs = data.refs if not context.event_freq_ref else generate_event_freq_ref(data.obs)
    outputs = MultiParOutput()
    for econ_par in context.econ_pars:
        curr_fcsts = probabilistic_to_deterministic_forecast(data.fcsts, econ_par)
        curr_refs = probabilistic_to_deterministic_forecast(refs, econ_par) # TODO: this one is possibly okay
        curr_data = InputData(data.obs, curr_fcsts, curr_refs)
        outputs.insert(econ_par, multiple_timesteps(econ_par, curr_data, context, parallel_nodes, verbose))
    
    return outputs


#
# TODO: move this to helpers.py
#
def probabilistic_to_deterministic_forecast(ensembles: np.ndarray, crit_thres: float) -> np.ndarray:
    if is_deterministic(ensembles[0]):
        raise ValueError('Cannot convert deterministic forecast to deterministic forecast')   
    return np.nanquantile(ensembles, 1 - crit_thres, axis=1)


# Can reproduce the behaviour of event frequency reference used in REV 
# using the RUV expected utility approach (optimisation over whole forecast
# distribution method) with an ensemble for each timestep
# which is simply the observation record. NA are dropped to simplify
# calculation of forecast likelihoods
#
# TODO: move this to helpers.py
#
def generate_event_freq_ref(obs: np.ndarray) -> np.ndarray:
    return np.tile(obs[~np.isnan(obs)], (obs.shape[0], 1))
