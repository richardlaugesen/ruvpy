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

from statsmodels.distributions.empirical_distribution import ECDF
import pytest

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *
from ruv.single_timestep import *
from ruv.data_classes import *


def test_ex_ante_utility():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    fast_economic_model = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})
    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = calc_likelihood(ens, thresholds)

    context = DecisionContext(None, damage_func, utility_func, thresholds, economic_model, fast_economic_model)

    alpha = 0.1
    spend = 0.052
    assert np.isclose(
        ex_ante_utility(alpha, spend, probs, context),
        -3.386, 1e-2)

    alpha = 0.7
    spend = 0.052
    assert np.isclose(
        ex_ante_utility(alpha, spend, probs, context),
        -3.386, 1e-2)

    alpha = 0.1
    spend = 3
    assert np.isclose(
        ex_ante_utility(alpha, spend, probs, context),
        -8.199, 1e-2)

    alpha = 0.1
    spend = 3
    thresholds = np.arange(1, 100000, 1)
    ens = np.random.normal(50000, 10000, 100000)    
    probs = calc_likelihood(ens, thresholds)   # tiny likelihoods
    context = DecisionContext(None, damage_func, utility_func, thresholds, economic_model, fast_economic_model)
    assert np.isclose(
        ex_ante_utility(alpha, spend, probs, context),
        -8.199, 1e-2)


def test_ex_post_utility():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    fast_economic_model = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    context = DecisionContext(None, damage_func, utility_func, thresholds, economic_model, fast_economic_model)

    occured = thresholds[10]
    alpha = 0.1
    spend = 0.052
    assert np.isclose(
        ex_post_utility(alpha, occured, spend, context),
        -3.386, 1e-2)

    occured = thresholds[10]
    alpha = 0.7
    spend = 0.052
    assert np.isclose(
        ex_post_utility(alpha, occured, spend, context),
        -3.847, 1e-2)

    occured = thresholds[10]
    alpha = 0.1
    spend = 3
    assert np.isclose(
        ex_post_utility(alpha, occured, spend, context),
        -8.199, 1e-2)


def test_find_spend_ensemble():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})
    context = DecisionContext(None, damage_func, utility_func, thresholds, economic_model, analytical_spend)

    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = calc_likelihood(ens, thresholds)
    alpha = 0.1
    assert np.isclose(find_spend_ensemble(alpha, ens, probs, context), 0.012, 1e-1)

    # Not implemented to work with deterministic forecasts so no need to test for it.
    # Code uses analytical_spend method of economic model instead of find_spend for 
    # deterministic forecasts. Noting here because it could be a source of hard to 
    # find bugs in the future.


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
    ref = np.random.normal(5, 3, 100)

    t = 0
    data = InputData([ob], [fcst], [ref])
    context = DecisionContext(None, damage_func, utility_func, thresholds, economic_model, analytical_spend)
    
    t, obs_spends, obs_ex_post, fcst_spends, fcst_ex_post, ref_spends, ref_ex_post = single_timestep(t, alpha, data, context)

    assert np.isclose(obs_spends, 0.0076, 1e-2)
    assert np.isclose(fcst_spends, 0.012, 1e-2)
    assert np.isclose(ref_spends, 0.0076, 1e-2)


def test_calc_likelihoods():
    np.random.seed(42)

    # typical ensemble and range of thresholds
    ens = np.random.normal(10, 1, 100)
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(calc_likelihood(ens, thresholds), np.array([0, 0, 0.01, 0.16, 0.37, 0.35, 0.11, 0, 0, 0]), 1e-1)

    # all in 1 class
    ens = np.random.normal(1000, 1, 100)
    thresholds = [0, 5]
    assert np.allclose(calc_likelihood(ens, thresholds), np.array([0, 1]), 1e-1)

    # adds to 1
    assert np.equal(np.sum(calc_likelihood(ens, thresholds)), 1)

    # Continuous decision with 100 member ensemble forecast
    assert np.array_equal(calc_likelihood(ens, None), np.full(100, 1e-2))

    # Calculating likelihoods with deterministic forecasts or forecasts with missing 
    # values is not implemented. No exception is raised if this is attempted though
    # and therefore not testing for it here. 
    # 
    # The function will never be called with deterministic forecasts according 
    # to the current code. 
    # 
    # Noting here because it could be a source of hard to find bugs in the future.

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

    assert np.isnan(realised_threshold(np.nan, thresholds))

