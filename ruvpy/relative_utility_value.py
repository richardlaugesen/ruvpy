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

from ruvpy.helpers import generate_event_freq_ref
from ruvpy.data_classes import DecisionContext


def relative_utility_value(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, decision_context: dict, parallel_nodes: int=4) -> dict:
    """
    Calculate the Relative Utility Value (RUV) for a set of observations, forecasts, and references
    using a specified decision-context.

    This function calculates RUV by evaluating the utility of forecasts through a simulation of decision-making
    under uncertainty. RUV quantifies the value of forecasts relative to a reference scenario (e.g., climatology)
    and is often applied in settings such as streamflow forecasting or weather forecasting. It works with
    probabilistic forecasts but can also handle deterministic forecasts (single ensemble member for `fcsts` and `refs`).

    The RUV method and RUVPY software package are introduced in the following publications:

    - Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range
      of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27,
      873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.
    - Laugesen, R., Thyer, M., McInerney, D., Kavetski, D. (2024). Software to quantify the value of forecasts for
      decision-making: sensitivity to damages case study. Manuscript submitted for publication in Environmental
      Modelling & Software.

    Args:
        obs (np.ndarray): 1D array of observed values representing the actual outcomes.
        fcsts (np.ndarray): 2D array of forecast values, where each column is an ensemble member and each row
            corresponds to a forecast at a given time.
        refs (np.ndarray): 2D array of reference values (e.g., climatology ensemble), where each column is an ensemble
            member and each row corresponds to a time period. If `None`, a reference climatology will be generated based
            on the observations, reproducing the observed frequency of events.
        decision_context (dict): Dictionary defining the decision context, containing:
            - 'decision_thresholds' (np.ndarray): 1D array specifying thresholds of the forecast variable.
            - 'decision_rule' (list): Decision-making function and a dictionary of its parameters.
            - 'damage_function' (list): Damage function method and a dictionary of its parameters.
            - 'utility_function' (list): Utility function method and a dictionary of its parameters.
            - 'economic_model' (list): Economic model function, analytical function, and list of parameter values.
        parallel_nodes (int, optional): Number of parallel processes used for computation. Defaults to 4.

    Returns:
        dict: Dictionary containing the calculated Relative Utility Value (RUV) results. Keys include:

            - 'ruv': 2D array of RUV values for each economic model parameter.
            - 'economic_model_params': Economic model parameters used in the calculation.
            - 'fcst_spends', 'ref_spends', 'obs_spends': Amount spent after decision optimization at each timestep.
            - 'fcst_ex_ante', 'ref_ex_ante', 'obs_ex_ante': Expected utility before the event occurred (ex ante).
            - 'fcst_ex_post', 'ref_ex_post', 'obs_ex_post': Utility after the event occurred (ex post).
            - 'avg_fcst_ex_post', 'avg_ref_ex_post', 'avg_obs_ex_post': Average ex post utility.

    Raises:
        ValueError: If inputs contain missing values, invalid thresholds are provided, or input data lengths do not match.

    Examples:
        Examples reproducing figures from the research papers can be found as Jupyter notebooks in the *examples* directory.

        An example decision context may help to understand the input structure required.

            decision_context = {
                'utility_function': [cara, {'A': 0.3}],
                'decision_rule': [optimise_over_forecast_distribution, None],
                'decision_thresholds': np.array([0, 5, 15, 25]),
                'economic_model': [cost_loss, cost_loss_analytical_spend, np.array([0.1, 0.5, 0.9])],
                'damage_function': [logistic, {'k': 0.2, 'A': 1, 'threshold': 20}]
            }

    Included decision context functions:

    Decision rules:

    - `optimise_over_forecast_distribution`: Optimises decision-making based on the whole forecast distribution.
    - `critical_probability_threshold_fixed`: Uses a fixed critical probability threshold for decision-making.
    - `critical_probability_threshold_max_value`: Selects the decision threshold leading to the maximum forecast value.
    - `critical_probability_threshold_equals_par`: Matches the decision threshold to the economic parameter.

    Damage functions:

    - `logistic`: Logistic damage function with defined maximum damages, steepness, and location.
    - `logistic_zero`: Logistic function with damages pegged to zero for zero flow.
    - `binary`: Binary loss function with parameters for max and min loss, and location.
    - `linear`: Linear damage function.
    - `user_defined`: Damage function interpolated over user-defined points.

    Utility functions:

    - `cara`: Constant Absolute Risk Aversion (CARA), where absolute risk aversion stays constant regardless of wealth.
    - `crra`: Constant Relative Risk Aversion (CRRA), where relative risk aversion stays constant regardless of wealth.
    - `exponential_utility`: Exponential utility function used to model CARA behaviour.
    - `isoelastic_utility`: Isoelastic utility function used to model CRRA behaviour.
    - `hyperbolic_utility`: Hyperbolic Absolute Risk Aversion (HARA), generalises both CARA and CRRA behavior.

    Economic models:

    - `cost_loss`: Standard cost-loss economic model based on spending to mitigate potential future losses.
    - `cost_loss_analytical_spend`: Analytical function to compute optimal spending in cost-loss.

    Decision types:

    Defined by providing a list of thresholds in the `decision_thresholds` key of the decision_context dictionary:

    - 'Binary decision': 1D array with two elements, 0 and the threshold value (e.g., np.array([0, 20])).
    - 'Multi-categorical decision': 1D array with multiple elements, one of which must be 0 (e.g., np.array([0, 5, 15, 25])).
    - 'Continuous decision': `None`.
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

