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


def setup():
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 0.5})
    decision_threshold = 0.3; 
    alpha = 0.1; 
    value = 0.7; 
    spend = 0.052
    economic_model = cost_loss
    utility_func = cara


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


def test_likelihoods():
    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(likelihoods(ens, thresholds),
                       np.array([0, 0, 0.01, 0.16, 0.37, 0.35, 0.11, 0, 0, 0]), 1e-1)

    with pytest.raises(ValueError):
        ens[20:60] = np.nan
        #ens = ens[~np.isnan(ens)]
        likelihoods(ens, thresholds)

    thresholds = None
    assert np.array_equal(likelihoods(ens, thresholds),
                          np.full(100, 1e-2))


def test_realised_threshold():
    thresholds = np.array([0, 3, 6])
    assert np.equal(realised_threshold(0.5, thresholds), 0)
    assert np.equal(realised_threshold(3, thresholds), 3)
    assert np.equal(realised_threshold(3.5, thresholds), 3)
    assert np.equal(realised_threshold(6, thresholds), 6)
    assert np.equal(realised_threshold(7, thresholds), 6)

    with pytest.raises(ValueError):
        values = [0.5, 3, 3.5, 6, 7]
        realised_threshold(values, thresholds)

    assert np.equal(realised_threshold(42, None), 42)


def test_probabilistic_to_deterministic_forecast():
    np.random.seed(42)
    fcst_ens = np.random.normal(10, 1, (1000, 5))  # (ens_members, timesteps)

    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([10.02788548, 9.96583783, 10.02239477, 10.06476768, 9.99314539]), 1e-5)
    
    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, -1)

    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, 2)

    fcst_ens = np.random.normal(10, 1, (1000, 1))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([9.95717293]), 1e-5)
    
    fcst_ens = np.random.normal(10, 1, (1, 10))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        fcst_ens[0], 1e-5)


def test_event_freq_ref():
    obs = np.random.gamma(1, 2, 1000)
    idx = np.random.randint(0, obs.size, 100)
    obs[idx] = np.nan
    ref = event_freq_ref(obs)
    assert np.isclose(np.mean(ref), np.nanmean(obs), 1e-5)
    assert np.array_equal(ref.shape, (obs.shape[0], obs.shape[0] - np.sum(np.isnan(obs))))
