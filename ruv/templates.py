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

from typing import Callable
import numpy as np

from ruv.multi_timestep import multiple_timesteps
from ruv.data_classes import MultiParOutput, DecisionContext
from ruv.helpers import probabilistic_to_deterministic_forecast


# ------------------------------------------------
# DECISION RULE
# ------------------------------------------------

def decision_rule_template(params: dict) -> Callable:

    # extract parameters from the dictionary if needed
    decision_rule_param_1 = params['decision_rule_param_1']
    # ...

    # define the actual decision rule
    def decision_rule(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:

        # do something with the parameters, perhaps adjust the forecast and reference
        fcsts = probabilistic_to_deterministic_forecast(fcsts, decision_rule_param_1)

        # create object to hold results from RUV implementation
        outputs = MultiParOutput()

        # loop over economic parameters
        for econ_par in context.economic_model_params:
            outputs.insert(econ_par, multiple_timesteps(obs, fcsts, refs, econ_par, context, parallel_nodes))
        return outputs

    return decision_rule


# ------------------------------------------------
# UTILITY FUNCTION
# ------------------------------------------------

def utility_function_template(params: dict) -> Callable:

    # extract parameters from the dictionary if needed
    utility_params_1 = params['utility_params_1']
    utility_params_2 = params['utility_params_2']
    # ...

    def utility(c: float) -> float:

        # use the parameters to adjust the outcome
        if utility_params_2 == 0:
            return c

        else:
            c = c * utility_params_1

    return utility


# ------------------------------------------------
# DAMAGE FUNCTION
# ------------------------------------------------

def damage_function_template(params: dict) -> Callable:

    # extract parameters from the dictionary if needed
    damage_function_param_1 = params['damage_function_param_1']
    damage_function_param_2 = params['damage_function_param_2']
    # ...

    # define the actual damage function
    def damage_function(magnitude: np.ndarray) -> np.ndarray:

        # do something with the parameters, perhaps adjust the magnitude
        magnitude = magnitude / damage_function_param_1

        return magnitude

    return damage_function

# ------------------------------------------------
# ECONOMIC MODEL
#
# Note for now it does not use the closure
# pattern as in the other components and only
# accepts single parameter economic models
# ------------------------------------------------

def economic_model_template(econ_param_1: float, values: np.ndarray, spend: float, damage_function: callable) -> np.ndarray:
    damages = damage_function(values)
    outcome = damages * econ_param_1 - spend
    return outcome


# fast analytical solution for spend, or return None
def economic_model_analytical_spend_template(econ_param_1: float, value: float, damage_function: callable) -> float:
    return None

