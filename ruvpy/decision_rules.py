from typing import Callable
from scipy.optimize import minimize_scalar
import numpy as np

from ruvpy.multi_timestep import multiple_timesteps
from ruvpy.data_classes import MultiParOutput, DecisionContext
from ruvpy.helpers import probabilistic_to_deterministic_forecast, nanmode


def optimise_over_forecast_distribution(params: dict) -> Callable:
    """Optimise spending over the entire forecast distribution."""
    # method has no params

    def decision_rule(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
        outputs = MultiParOutput()
        for econ_par in context.economic_model_params:
            outputs.insert(econ_par, multiple_timesteps(obs, fcsts, refs, econ_par, context, parallel_nodes))
        return outputs

    return decision_rule


def critical_probability_threshold_fixed(params: dict) -> Callable:
    """Use a fixed critical probability threshold."""
    crit_prob_thres = params['critical_probability_threshold']

    def decision_rule(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
        fcsts = probabilistic_to_deterministic_forecast(fcsts, crit_prob_thres)
        outputs = MultiParOutput()
        for econ_par in context.economic_model_params:
            outputs.insert(econ_par, multiple_timesteps(obs, fcsts, refs, econ_par, context, parallel_nodes))
        return outputs

    return decision_rule


def critical_probability_threshold_max_value(params: dict) -> Callable:
    """Search for the critical probability that maximises RUV."""
    # method has no params

    def decision_rule(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
        outputs = MultiParOutput()
        for econ_par in context.economic_model_params:
            def minimise_this(crit_prob_thres):
                curr_fcsts = probabilistic_to_deterministic_forecast(fcsts, crit_prob_thres)
                return -multiple_timesteps(obs, curr_fcsts, refs, econ_par, context, parallel_nodes).ruv
            max_ruv_thres = minimize_scalar(minimise_this, method='bounded', bounds=(0, 1), options={'disp': False, 'xatol': 0.005}).x

            max_fcsts = probabilistic_to_deterministic_forecast(fcsts, max_ruv_thres)
            outputs.insert(econ_par, multiple_timesteps(obs, max_fcsts, refs, econ_par, context, parallel_nodes))
        return outputs

    return decision_rule


def critical_probability_threshold_equals_par(params: dict) -> Callable:
    """Set the critical probability equal to the economic parameter."""
    # method has no params

    def decision_rule(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
        outputs = MultiParOutput()
        for econ_par in context.economic_model_params:
            curr_fcsts = probabilistic_to_deterministic_forecast(fcsts, econ_par)
            outputs.insert(econ_par, multiple_timesteps(obs, curr_fcsts, refs, econ_par, context, parallel_nodes))
        return outputs

    return decision_rule


def forecast_distribution_mode(params: dict) -> Callable:
    """Use the mode of the forecast distribution as the decision variable."""
    # method has no params

    def decision_rule(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, context: DecisionContext, parallel_nodes: int) -> MultiParOutput:
        outputs = MultiParOutput()
        curr_fcsts = nanmode(fcsts, axis=1)
        for econ_par in context.economic_model_params:
            outputs.insert(econ_par, multiple_timesteps(obs, curr_fcsts, refs, econ_par, context, parallel_nodes))
        return outputs

    return decision_rule
