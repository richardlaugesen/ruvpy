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

import numpy as np

from ruv.data_classes import DecisionContext
from ruv.economic_models import cost_loss, cost_loss_analytical_spend
from ruv.utility_functions import cara
from ruv.damage_functions import logistic_zero
from ruv.helpers import generate_event_freq_ref
from ruv.decision_rules import optimise_over_forecast_distribution, critical_probability_threshold_equals_par, critical_probability_threshold_fixed, critical_probability_threshold_max_value


def get_data(ref_equals_fcst=False, event_freq_ref=False):
    np.random.seed(42)

    fcsts = np.random.normal(10, 1, (20, 100))  # (timesteps, ens_members)
    fcsts[fcsts < 0] = 0

    obs = np.random.gamma(1, 5, (20, 1))
    obs[obs < 0] = 0

    if ref_equals_fcst:
        refs = fcsts
    else:
        refs = np.random.normal(5, 3, (20, 100))
        refs[refs < 0] = 0

        if event_freq_ref:
            refs = generate_event_freq_ref(obs)

    return obs, fcsts, refs


def get_context(decision_rule, decision_rule_params, risk_aversion=0.3):
    context_params = {
        'economic_model_params': np.array([0.001, 0.25, 0.5, 0.75, 0.999]),
        'utility_function': cara({'A': risk_aversion}),
        'damage_function': logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15}),
        'decision_thresholds': np.arange(0, 20, 3),
        'economic_model': cost_loss,
        'analytical_spend': cost_loss_analytical_spend,
        'decision_rule': decision_rule(decision_rule_params)
    }
    return DecisionContext(**context_params)


def test_optimise_over_forecast_distribution():
    context = get_context(optimise_over_forecast_distribution, None)

    # basic ensemble fcst and ref
    obs, fcsts, refs = get_data()
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [-0.001065, -0.180943, -0.825034, -2.662902, -197.032082], 1e-3)

    # event freq ref
    obs, fcsts, refs = get_data(event_freq_ref=True)
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [-35.1915, -0.1809, -0.8581, -2.8990, -197.0321], 1e-3)

    # ref equals fcst
    obs, fcsts, refs = get_data(ref_equals_fcst=True)
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 1e-3)


def test_critical_probability_threshold_equals_par():
    obs, fcsts, refs = get_data()

    context = get_context(critical_probability_threshold_equals_par, None, 0)
    econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    context = get_context(optimise_over_forecast_distribution, None, 0)
    optim_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(econ_par_result.get_series('ruv'), optim_result.get_series('ruv'), 1e-3)

    context = get_context(critical_probability_threshold_equals_par, None, 0.1)
    econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    context = get_context(optimise_over_forecast_distribution, None, 0.1)
    optim_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(econ_par_result.get_series('ruv'), optim_result.get_series('ruv'), 1e-3)

    context = get_context(critical_probability_threshold_equals_par, None, 5)
    econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    context = get_context(optimise_over_forecast_distribution, None, 5)
    optim_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert not np.allclose(econ_par_result.get_series('ruv'), optim_result.get_series('ruv'), 1e-3)


def test_critical_probability_threshold_fixed():
    obs, fcsts, refs = get_data()
    context = get_context(critical_probability_threshold_fixed, {'critical_probability_threshold': 0.5})
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [-0.8561, -0.1809, -0.8250, -2.6629, -1011.4097], 1e-3)


def test_critical_probability_threshold_max_value():
    obs, fcsts, refs = get_data()
    context = get_context(critical_probability_threshold_max_value, None)
    max_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(max_result.get_series('ruv'), [-0.8561, 0.0000, -0.1039, -0.4304, -197.0321], 1e-3)

    context = get_context(critical_probability_threshold_equals_par, None)
    econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.all(max_result.get_series('ruv')[1:] >= econ_par_result.get_series('ruv')[1:])   # ignore first value because economic parameter value is extremely small

