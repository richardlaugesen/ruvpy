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

import pytest
import numpy as np

from ruv.decision_methods import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.damage_functions import *


def test_probabilistic_to_deterministic_forecast():
    np.random.seed(42)
    # 1000 member ensembles for 5 timesteps
    fcst_ens = np.random.normal(10, 1, (5, 1000))

    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([10.02530061, 10.06307713,  9.99974924, 10.00018457,  9.98175801]), 1e-5)

    assert np.alltrue(probabilistic_to_deterministic_forecast(
        fcst_ens, 0.1) > probabilistic_to_deterministic_forecast(fcst_ens, 0.9))

    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, -1)

    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, 2)

    # 1000 member ensemble for a single timestep
    fcst_ens = np.random.normal(10, 1, (1, 1000))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([9.957]), 1e-3)

    # deterministic forecast for 10 timesteps
    with pytest.raises(ValueError):
        fcst_ens = np.random.normal(10, 1, (10, 1))
        assert np.array_equal(
            probabilistic_to_deterministic_forecast(fcst_ens, 0.5), fcst_ens, 1e-5)

    # deterministic observations for 1000 timesteps
    with pytest.raises(ValueError):
        obs = np.random.gamma(1, 2, 1000)
        assert np.array_equal(
            probabilistic_to_deterministic_forecast(obs, 0.25), obs)


def test_generate_event_freq_ref():
    assert np.array_equal(
        generate_event_freq_ref(np.array([6, 7, 3, np.nan, 2])),
        np.array([[6, 7, 3, 2], [6, 7, 3, 2], [6, 7, 3, 2], [6, 7, 3, 2], [6, 7, 3, 2]]))

    obs = np.random.gamma(1, 2, 1000)
    idx = np.random.randint(0, obs.size, 100)
    obs[idx] = np.nan
    ref = generate_event_freq_ref(obs)
    assert np.isclose(np.mean(ref), np.nanmean(obs), 1e-5)
    assert np.array_equal(
        ref.shape, (obs.shape[0], obs.shape[0] - np.sum(np.isnan(obs))))


def test_optimise_over_forecast_distribution():    
    
    # basic ensemble fcst and ref
    data = get_data()
    context = get_context()
    result = optimise_over_forecast_distribution(data, context, 1)
    assert np.allclose(result.get_series('ruv'), [0.3101, -0.22249, -1.093606, -3.50433, -262.69065], 1e-3)

    # event freq ref
    context = get_context(event_freq_ref=True)
    result = optimise_over_forecast_distribution(data, context, 1)
    assert np.allclose(result.get_series('ruv'), [-34.6425, -0.265098, -1.1084778, -3.67724, -262.69065], 1e-3)

    # ref equals fcst
    data = get_data(ref_equals_fcst=True)
    context = get_context()
    result = optimise_over_forecast_distribution(data, context, 1)
    assert np.allclose(result.get_series('ruv'), [0, 0, 0, 0, 0], 1e-3)


def test_critical_probability_threshold_equals_alpha():
    data = get_data()
    context = get_context(risk_aversion=0.1)
    alpha_result = critical_probability_threshold_equals_alpha(data, context, 1)
    optim_result = optimise_over_forecast_distribution(data, context, 1)
    assert np.allclose(alpha_result.get_series('ruv'), optim_result.get_series('ruv'), 1e-3)

    data = get_data()
    context = get_context(risk_aversion=5)
    alpha_result = critical_probability_threshold_equals_alpha(data, context, 1)
    optim_result = optimise_over_forecast_distribution(data, context, 1)
    assert not np.allclose(alpha_result.get_series('ruv'), optim_result.get_series('ruv'), 1e-3)


def test_critical_probability_threshold_fixed():
    data = get_data()
    context = get_context(crit_prob_thres=0.5)
    result = critical_probability_threshold_fixed(data, context, 1)
    assert np.allclose(result.get_series('ruv'), [0.00398, -0.22249, -1.093606, -3.50433, -1271.97656], 1e-3)


def test_critical_probability_threshold_max_value():
    data = get_data()
    context = get_context()
    max_result = critical_probability_threshold_max_value(data, context, 1)
    assert np.allclose(max_result.get_series('ruv'), [0.00398, 0, -0.18472, -0.66092, -262.6907], 1e-3)

    alpha_result = critical_probability_threshold_equals_alpha(data, context, 1)
    assert np.alltrue(max_result.get_series('ruv')[1:] >= alpha_result.get_series('ruv')[1:])   # ignore first value because alpha value is extremely small


def get_data(ref_equals_fcst=False):    
    np.random.seed(42)
    fcsts = np.random.normal(10, 1, (20, 100))  # (timesteps, ens_members)
    fcsts[fcsts < 0] = 0
    refs = np.random.normal(5, 3, (20, 100)) if not ref_equals_fcst else fcsts
    refs[refs < 0] = 0
    obs = np.random.gamma(1, 5, (20, 1))
    obs[obs < 0] = 0    
    return InputData(obs, fcsts, refs)


def get_context(event_freq_ref=False, crit_prob_thres=None, risk_aversion=0.3):
    alphas = np.array([0.001, 0.25, 0.5, 0.75, 0.999])
    thresholds = np.arange(0, 20, 3)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': risk_aversion})
    return DecisionContext(alphas, damage_func, utility_func, thresholds, economic_model, analytical_spend, crit_prob_thres=crit_prob_thres, event_freq_ref=event_freq_ref)
