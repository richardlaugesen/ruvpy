from typing import Callable
import numpy as np

from ruvpy.multi_timestep import multiple_timesteps
from ruvpy.data_classes import MultiParOutput, DecisionContext
from ruvpy.helpers import probabilistic_to_deterministic_forecast


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

