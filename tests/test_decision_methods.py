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


# TODO: Move all this into seperate test functions for each of the different decision making methods
# def test_multiple_alpha():

#     alphas = np.array([0.001, 0.25, 0.5, 0.75, 0.999])
#     thresholds = np.arange(0, 20, 3)
#     economic_model = cost_loss
#     analytical_spend = cost_loss_analytical_spend
#     damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
#     utility_func = cara({'A': 0.3})

#     np.random.seed(42)
#     num_steps = 20
#     ens_size = 100

#     # (timesteps, ens_members)
#     obs = np.random.gamma(1, 5, (num_steps, 1))
#     obs[obs < 0] = 0

#     fcsts = np.random.normal(10, 1, (num_steps, ens_size))
#     fcsts[fcsts < 0] = 0

#     fcst_likelihoods = all_likelihoods(obs, fcsts, thresholds)
#     refs = np.random.normal(5, 3, (num_steps, ens_size))
#     refs[refs < 0] = 0
#     ref_likelihoods = all_likelihoods(obs, refs, thresholds)

#     obs_likelihoods = all_likelihoods(obs, obs, thresholds)

#     decision_method = 'optimise_over_forecast_distribution'
#     result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     assert np.allclose(
#         result['ruv'],
#         [0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)

#     decision_method = 'critical_probability_threshold_equals_alpha'
#     result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     assert np.allclose(
#         result['ruv'],
#         [0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)

#     refs = generate_event_freq_ref(obs)
#     ref_likelihoods = all_likelihoods(obs, refs, thresholds)

#     decision_method = 'optimise_over_forecast_distribution'
#     result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     assert np.allclose(
#         result['ruv'],
#         [-74.0584681, -0.0742971679, -0.472369878, -1.71864364, -117.108684], 1e-3)

#     decision_method = 'critical_probability_threshold_equals_alpha'
#     result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     assert np.allclose(
#         result['ruv'],
#         [-74.0584681, -0.0742971679, -0.472369878, -1.71864364, -117.108684], 1e-3)

#     # two methods are equivilent when risk aversion is small
#     utility_func = cara({'A': 0.1})
#     decision_method = 'optimise_over_forecast_distribution'
#     result_1 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     result_1 = result_1
#     decision_method = 'critical_probability_threshold_equals_alpha'
#     result_2 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     assert np.allclose(result_1['ruv'], result_2['ruv'], 1e-3)

#     # two methods not equivilent when risk aversion is high
#     utility_func = cara({'A': 5})
#     decision_method = 'optimise_over_forecast_distribution'
#     result_1 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     decision_method = 'critical_probability_threshold_equals_alpha'
#     result_2 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, obs_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
#     assert not np.allclose(result_1['ruv'], result_2['ruv'], 1e-3)
