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
def relative_utility_value(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, decision_context: dict, parallel_nodes: int=4) -> dict:
    """
    Calculate the Relative Utility Value (RUV) for a set of observations, forecasts, and references using
    specified decision-context.

    This entry function to RUVPY calculates RUV by evaluating the utility of forecasts through a simulation of decision-making under uncertainty.
    RUV quantifies the value of forecasts relative to a reference scenario (e.g., climatology) and is often applied in settings such as
    streamflow forecasting or weather forecasting, where decisions are made based on forecast probabilities.
    It is typically used with probabilistic forecasts but can also handle deterministic forecasts (single ensemble member for fcsts and refs).

    The RUV method and RUVPY software package are introduced in detail in the following publications. We suggest reading these to understand the context and motivation for the software.

        **Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27, 873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.**

        **Laugesen, R., Thyer, M., McInerney, D., Kavetski, D. (2024). Software to quantify the value of forecasts for decision-making: sensitivity to damages case study. Manuscript submitted for publication in Environmental Modelling & Software.**

    Parameters
    ----------
    obs : np.ndarray
        1D array of observed values, representing the actual outcomes.
    fcsts : np.ndarray
        2D array of forecast values, where each column is an ensemble member and each row corresponds to a forecast at a given time.
    refs : np.ndarray
        2D array of reference values (e.g., climatology ensemble), where each column is an ensemble member and each row corresponds to a time period.
        If None, a reference climatology will be generated based on the observations, reproduces the observed frequency of events as in REV.
    decision_context : dict
        Dictionary defining the decision-context. The dictionary must contain:
        - 'decision_thresholds': A 1D array specifying thresholds used to define decisions based on forecast values.
        - 'decision_rule': A list with the decision-making function as the first element, and its parameters as the second.
        - 'damage_function': A list with the damage function method as the first element, and its parameters as the second.
        - 'utility_function': A list with the utility function method as the first element, and its parameters as the second.
        - 'economic_model': A list with the economic model function, its analytical spend function, and a list of parameter values (e.g. alpha).
    parallel_nodes : int, optional
        The number of parallel processes used for computation. Defaults to 4.

    Returns
    -------
    dict
        Dictionary containing the calculated Relative Utility Value (RUV) results for the given forecasts.
        The dictionary includes:
        - 'ruv': A 2D array of RUV values for each economic model parameter.
        - 'economic_model_params': The economic model parameters used in the calculation for convenience.
        - 'avg_fcst_ex_post', 'avg_ref_ex_post', 'avg_obs_ex_post': Average ex post results for forecasts, references, and observations, respectively.
        - 'fcst_spends', 'ref_spends', 'obs_spends': Spending values for forecasts, references, and observations.
        - 'fcst_ex_ante', 'ref_ex_ante', 'obs_ex_ante': Ex ante (expected) results for forecasts, references, and observations.
        - 'fcst_ex_post', 'ref_ex_post', 'obs_ex_post': Ex post (actual) results for forecasts, references, and observations.

    Raises
    ------
    ValueError
        If inputs contain missing values, or if thresholds are violated (e.g., forecast or observation values are less than the smallest decision threshold).
        Also raised if the lengths of `obs`, `fcsts`, and `refs` do not match.

    Examples
    --------
    Example decision context:

        decision_context = {
            'utility_function': [cara, {'A': 0.3}],
            'decision_rule': [optimise_over_forecast_distribution, None],
            'decision_thresholds': np.array([0, 5, 15, 25]),
            'economic_model': [cost_loss, cost_loss_analytical_spend, np.array([0.1, 0.5, 0.9])],
            'damage_function': [logistic, {'k': 0.2, 'A': 1, 'threshold': 20}]
        }

    Complete examples reproducing figures from related research are included as Jupyter notebooks in the *examples* directory.

    Included decision context functions
    ----------------------------------------

    **Decision rules**:
    - `optimise_over_forecast_distribution`: Optimises decision-making based on whole forecast distribution.
    - `critical_probability_threshold_fixed`: Uses a fixed critical probability threshold for decision-making.
    - `critical_probability_threshold_max_value`: Selects the decision threshold which leads to the maximum forecast value.
    - `critical_probability_threshold_equals_par`: Matches the decision threshold to the economic parameter.

    **Damage functions**:
    - `logistic`: A logistic damage function with a defined maximium damages, steepness, and location.
    - `logistic_zero`: A logistic function with damages pegged to zero for zero flow.
    - `binary`: A binary loss function with parameters for max and min loss, and location.
    - `linear`: A linear damage function.
    - `user_defined`: A damage function interpolated over user-defined points.

    **Utility functions**:
    - `cara`: Constant Absolute Risk Aversion (CARA), where absolute risk aversion stays constant regardless of wealth.
    - `crra`: Constant Relative Risk Aversion (CRRA), where relative risk aversion stays constant regardless of wealth.
    - `exponential_utility`: Exponential utility function used to model CARA behaviour.
    - `isoelastic_utility`: Isoelastic utility function used to model CRRA behaviour.
    - `hyperbolic_utility`: Hyperbolic Absolute Risk Aversion (HARA), generalises both CARA and CRRA behaviour.

    **Economic models**:
    - `cost_loss`: Standard cost-loss economic model based on spending to mitigate potential future losses.
    - `cost_loss_analytical_spend`: An analytical function to compute optimal spending in cost-loss.

    **Decision types**:
    Defined by providing a list of thresholds in the `decision_thresholds` key of the decision_context dictionary
    - Binary decision: 1D array with two elements, 0 and the threshold value (e.g., np.array([0, 20]))
    - Multi-categorical decision: 1D array with multiple elements, one must be 0 (e.g., np.array([0, 5, 15, 25]))
    - Continuous decision: None
    """

    # decision type
    decision_thresholds = decision_context['decision_thresholds']

    # decision-making method
    decision_rule = decision_context['decision_rule'][0]
    decision_rule_params = decision_context['decision_rule'][1]
    decision_rule_fnc = decision_rule(decision_rule_params)

    # damage function
    damage_fnc_mth = decision_context['damage_function'][0]
    damage_fnc_params = decision_context['damage_function'][1]
    damage_function = damage_fnc_mth(damage_fnc_params)

    # utility function
    utility_fnc_mth = decision_context['utility_function'][0]
    utility_fnc_params = decision_context['utility_function'][1]
    utility_function = utility_fnc_mth(utility_fnc_params)

    # economic model
    economic_model_fnc = decision_context['economic_model'][0]
    economic_model_analytical_spend_fnc = decision_context['economic_model'][1]
    economic_model_params = decision_context['economic_model'][2]

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

