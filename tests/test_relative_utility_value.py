# Copyright 2023 Richard Laugesen
#
# This file is part of RUV
#
# RUV is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RUV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RUV.  If not, see <https://www.gnu.org/licenses/>.

from statsmodels.distributions.empirical_distribution import ECDF
import pytest

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *


def test_ecdf_numpy():
    ens = np.array([])
    thresholds = np.array([])
    assert np.array_equal(ecdf_numpy(ens, thresholds), np.array([]))

    ens = np.array([1, 2, 3, 4, 5])
    thresholds = np.array([])
    assert np.array_equal(ecdf_numpy(ens, thresholds), np.array([]))

    ens = np.array([])
    thresholds = np.array([1, 3, 5])
    assert np.all(np.isnan(ecdf_numpy(ens, thresholds)))

    ens = np.array([1, 2, 3, 4, 5])
    thresholds = np.array([0, 3])
    assert np.array_equal(ecdf_numpy(ens, thresholds), np.array([1, 0.6]))

    ens = np.random.normal(10, 1, 1000)
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(ecdf_numpy(ens, thresholds),
                      np.subtract(1, ECDF(ens, 'left')(thresholds)), 1e-3)


def test_calc_likelihoods():
    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(calc_likelihoods(ens, thresholds),
                       np.array([0, 0, 0.01, 0.16, 0.37, 0.35, 0.11, 0, 0, 0]), 1e-1)

    with pytest.raises(ValueError):
        calc_likelihoods(3, thresholds)

    with pytest.raises(ValueError):
        calc_likelihoods(3.2, thresholds)

    with pytest.raises(ValueError):
        calc_likelihoods([3], thresholds)

    with pytest.raises(ValueError):
        ens[20:60] = np.nan
        #ens = ens[~np.isnan(ens)]
        calc_likelihoods(ens, thresholds)

    thresholds = None
    assert np.array_equal(calc_likelihoods(ens, thresholds),
                          np.full(100, 1e-2))
    

def test_realised_threshold():
    thresholds = np.array([0, 3, 6])
    assert np.equal(realised_threshold(0.5, thresholds), 0)
    assert np.equal(realised_threshold(3, thresholds), 3)
    assert np.equal(realised_threshold(3.5, thresholds), 3)
    assert np.equal(realised_threshold(6, thresholds), 6)
    assert np.equal(realised_threshold(7, thresholds), 6)

    with pytest.raises(ValueError):
        realised_threshold(0.1, [1, 2, 3])

    with pytest.raises(ValueError):
        values = [0.5, 3, 3.5, 6, 7]
        realised_threshold(values, thresholds)

    assert np.equal(realised_threshold(42, None), 42)


def test_probabilistic_to_deterministic_forecast():
    np.random.seed(42)
    fcst_ens = np.random.normal(10, 1, (5, 1000))  # (timesteps, ens_members)

    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([10.02530061, 10.06307713,  9.99974924, 10.00018457,  9.98175801]), 1e-5)
    
    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, -1)

    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, 2)

    fcst_ens = np.random.normal(10, 1, (1, 1000))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([9.95717293]), 1e-5)
    
    fcst_ens = np.random.normal(10, 1, (10, 1))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        fcst_ens.T, 1e-5)


def test_event_freq_ref():
    obs = np.random.gamma(1, 2, 1000)
    idx = np.random.randint(0, obs.size, 100)
    obs[idx] = np.nan
    ref = event_freq_ref(obs)
    assert np.isclose(np.mean(ref), np.nanmean(obs), 1e-5)
    assert np.array_equal(ref.shape, (obs.shape[0], obs.shape[0] - np.sum(np.isnan(obs))))


def test_ex_ante_utility():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})
    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = calc_likelihoods(ens, thresholds)

    alpha = 0.1; spend = 0.052
    assert np.isclose(
        ex_ante_utility(spend, probs, thresholds, alpha, economic_model, damage_func, utility_func),
        -3.386, 1e-2)
    
    alpha = 0.7; spend = 0.052
    assert np.isclose(
        ex_ante_utility(spend, probs, thresholds, alpha, economic_model, damage_func, utility_func),
        -3.386, 1e-2)
    
    alpha = 0.1; spend = 3
    assert np.isclose(
        ex_ante_utility(spend, probs, thresholds, alpha, economic_model, damage_func, utility_func),
        -8.199, 1e-2)    


def test_ex_post_utility():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})
    
    occured = thresholds[10]; alpha = 0.1; spend = 0.052
    assert np.isclose(
        ex_post_utility(occured, spend, alpha, economic_model, damage_func, utility_func),
        -3.386, 1e-2)
    
    occured = thresholds[10]; alpha = 0.7; spend = 0.052
    assert np.isclose(
        ex_post_utility(occured, spend, alpha, economic_model, damage_func, utility_func),
        -3.847, 1e-2)
    
    occured = thresholds[10]; alpha = 0.1; spend = 3
    assert np.isclose(
        ex_post_utility(occured, spend, alpha, economic_model, damage_func, utility_func),
        -8.199, 1e-2)    


def test_find_spend():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = calc_likelihoods(ens, thresholds)
    alpha = 0.1
    assert np.isclose(
        find_spend(ens, probs, thresholds, alpha, economic_model, analytical_spend, damage_func, utility_func),
        0.012, 1e-1)

    det = 7.2
    probs = None
    alpha = 0.1
    assert np.isclose(
        find_spend(det, probs, thresholds, alpha, economic_model, analytical_spend, damage_func, utility_func),
        0.0018, 1e-3)


def test_single_timestep():
    t = 1
    ob = 10
    alpha = 0.1    
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)    
    fcst = np.random.normal(10, 1, 100)
    fcst_probs = calc_likelihoods(fcst, thresholds)

    ref = np.random.normal(5, 3, 100)
    ref_probs = calc_likelihoods(ref, thresholds)

    t, obs_spends, obs_ex_post, fcst_spends, fcst_ex_post, ref_spends, ref_ex_post = single_timestep(t, ob, thresholds, alpha, economic_model, analytical_spend, damage_func, utility_func, fcst, fcst_probs, ref, ref_probs)
    assert np.isclose(obs_spends, 0.0076, 1e-2) 
    assert np.isclose(fcst_spends, 0.012, 1e-2) 
    assert np.isclose(ref_spends, 0.0076, 1e-2) 


def test_multiple_timesteps():
    alpha = 0.1
    thresholds = np.arange(0, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)
    num_steps = 20
    ens_size = 100

    # (timesteps, ens_members)
    obs = np.random.gamma(1, 5, (num_steps, 1))
    fcst = np.random.normal(10, 1, (num_steps, ens_size))
    fcst_likelihoods = all_likelihoods(obs, fcst, thresholds)
    ref = np.random.normal(5, 3, (num_steps, ens_size))
    ref_likelihoods = all_likelihoods(obs, ref, thresholds)

    result = multiple_timesteps(alpha, obs, fcst, ref, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, 1, False)

    assert np.isclose(result[0], 0.0445, 1e-2)
    assert np.isclose(result[1], -3.399, 1e-2)
    assert np.isclose(result[2], -3.340, 1e-2)
    assert np.isclose(result[3], -3.402, 1e-2)
