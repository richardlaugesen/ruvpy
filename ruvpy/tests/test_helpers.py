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

from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import pytest

from ruvpy.helpers import probabilistic_to_deterministic_forecast, generate_event_freq_ref, ecdf, is_deterministic, risk_premium_to_prob_premium, risk_aversion_coef_to_risk_premium, risk_premium_to_risk_aversion_coef, prob_premium_to_risk_aversion_coef


def test_probabilistic_to_deterministic_forecast():
    np.random.seed(42)
    # 1000 member ensembles for 5 timesteps
    fcst_ens = np.random.normal(10, 1, (5, 1000))

    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([10.02530061, 10.06307713,  9.99974924, 10.00018457,  9.98175801]), 1e-5)

    assert np.all(probabilistic_to_deterministic_forecast(
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


def test_ecdf():
    ens = np.array([])
    thresholds = np.array([])
    assert np.array_equal(ecdf(ens, thresholds), np.array([]))

    ens = np.array([1, 2, 3, 4, 5])
    thresholds = np.array([])
    assert np.array_equal(ecdf(ens, thresholds), np.array([]))

    ens = np.array([])
    thresholds = np.array([1, 3, 5])
    assert np.all(np.isnan(ecdf(ens, thresholds)))

    ens = np.array([1, 2, 3, 4, 5])
    thresholds = np.array([0, 3])
    assert np.array_equal(ecdf(ens, thresholds), np.array([1, 0.6]))

    ens = np.random.normal(10, 1, 1000)
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(ecdf(ens, thresholds),
                       np.subtract(1, ECDF(ens, 'left')(thresholds)), 1e-3)

    assert np.array_equal(ecdf([5], thresholds), ecdf(np.array([5]), thresholds))


def test_is_deterministic_timestep():
    np.random.seed(42)

    # 1000 member ensembles for 5 timesteps
    ens = np.random.normal(10, 1, (5, 1000))
    with pytest.raises(ValueError):
        is_deterministic(ens)

    # deterministic forecast for 10 timesteps
    ens = np.random.normal(10, 1, (10, 1))
    with pytest.raises(ValueError):
        is_deterministic(ens)

    # single timestep of 1000 member ensemble
    ens = np.random.normal(10, 1, (1, 1000))
    with pytest.raises(ValueError):
        is_deterministic(ens)

    ens = np.random.gamma(1, 2, 1000)   # ensemble with 1000 members
    assert not is_deterministic(ens)

    ob = 5   # single value
    assert is_deterministic(ob)

    obs = np.array([5])   # array with single value
    assert is_deterministic(obs)


def test_risk_aversion_coef_to_risk_premium():
    risk_aversions = [0.3, 1, 5]
    gamble_size = 1
    expected_result = [0.1478, 0.4338, 0.8614]
    for i, risk_aversion in enumerate(risk_aversions):
        assert np.isclose(risk_aversion_coef_to_risk_premium(
            risk_aversion, gamble_size), expected_result[i], 1e-3)

    risk_aversions = [0.3, 1, 5]
    gamble_size = 10
    expected_result = [0.7698, 0.9307, 0.98614]
    for i, risk_aversion in enumerate(risk_aversions):
        assert np.isclose(risk_aversion_coef_to_risk_premium(
            risk_aversion, gamble_size), expected_result[i], 1e-3)


def test_risk_premium_to_risk_aversion_coef():
    risk_premiums = [0.1478, 0.4338, 0.8614]
    gamble_size = 1
    expected_result = [0.3, 1, 5]
    for i, risk_premium in enumerate(risk_premiums):
        assert np.isclose(risk_premium_to_risk_aversion_coef(
            risk_premium, gamble_size), expected_result[i], 1e-3)

    risk_premiums = [0.7698, 0.9307, 0.98614]
    gamble_size = 10
    expected_result = [0.3, 1, 5]
    for i, risk_premium in enumerate(risk_premiums):
        assert np.isclose(risk_premium_to_risk_aversion_coef(
            risk_premium, gamble_size), expected_result[i], 1e-3)


def test_risk_premium_to_prob_premium():
    risk_premiums = [0.1478, 0.4338, 0.8614]
    expected_result = [0.0744, 0.2311, 0.4933]
    for i, risk_premium in enumerate(risk_premiums):
        assert np.isclose(risk_premium_to_prob_premium(
            risk_premium), expected_result[i], 1e-3)

    risk_premium_probs = [0.283559, 0.662501, 0.930685]
    expected_result = [0.145656, 0.380797, 0.499955]
    for i, risk_premium_prob in enumerate(risk_premium_probs):
        assert np.isclose(risk_premium_to_prob_premium(
            risk_premium_prob), expected_result[i], 1e-3)

    risk_premium_prob = 0.7698
    expected_result = 0.452574
    assert np.isclose(risk_premium_to_prob_premium(
        risk_premium_prob), expected_result, 1e-3)

    risk_premium_prob = 0.9999
    with pytest.raises(Exception):
        risk_premium_to_prob_premium(risk_premium_prob)


def test_prob_premium_to_risk_aversion_coef():
    risk_premium_probs = [0.0744, 0.2311, 0.4933]
    gamble_size = 1
    expected_result = [0.3, 1, 5]
    for i, risk_premium_prob in enumerate(risk_premium_probs):
        assert np.isclose(prob_premium_to_risk_aversion_coef(
            risk_premium_prob, gamble_size), expected_result[i], 1e-3)

    risk_premium_probs = [0.145656, 0.380797, 0.499955]
    gamble_size = 2
    expected_result = [0.3, 1, 5]
    for i, risk_premium_prob in enumerate(risk_premium_probs):
        assert np.isclose(prob_premium_to_risk_aversion_coef(
            risk_premium_prob, gamble_size), expected_result[i], 1e-3)

    risk_premium_prob = 0.93
    gamble_size = 10
    with pytest.raises(Exception):
        prob_premium_to_risk_aversion_coef(risk_premium_prob, gamble_size)
