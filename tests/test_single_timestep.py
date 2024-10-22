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

import pytest
import numpy as np

from ruvpy.single_timestep import _calc_likelihood, _ex_ante_utility, _ex_post_utility, _find_spend_ensemble, single_timestep, _realised_threshold
from ruvpy.damage_functions import logistic_zero
from ruvpy.economic_models import cost_loss, cost_loss_analytical_spend
from ruvpy.utility_functions import cara
from ruvpy.data_classes import DecisionContext
from ruvpy.decision_rules import optimise_over_forecast_distribution


def get_context(decision_thresholds=np.arange(5, 20, 1)):
    context_fields = {
        'economic_model_params': None,
        'damage_function': logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15}),
        'utility_function': cara({'A': 0.3}),
        'decision_thresholds': decision_thresholds,
        'economic_model': cost_loss,
        'analytical_spend': cost_loss_analytical_spend,
        'decision_rule': optimise_over_forecast_distribution
    }
    return DecisionContext(**context_fields)


def test_ex_ante_utility():
    context = get_context()

    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = _calc_likelihood(ens, context.decision_thresholds)

    econ_par = 0.1
    spend = 0.052
    assert np.isclose(
        _ex_ante_utility(econ_par, spend, probs, context),
        -3.386, 1e-2)

    econ_par = 0.7
    spend = 0.052
    assert np.isclose(
        _ex_ante_utility(econ_par, spend, probs, context),
        -3.386, 1e-2)

    econ_par = 0.1
    spend = 3
    assert np.isclose(
        _ex_ante_utility(econ_par, spend, probs, context),
        -8.199, 1e-2)

    econ_par = 0.1
    spend = 3

    context = get_context(np.arange(1, 100000, 1))
    ens = np.random.normal(50000, 10000, 100000)    
    probs = _calc_likelihood(ens, context.decision_thresholds)   # tiny likelihoods
    assert np.isclose(
        _ex_ante_utility(econ_par, spend, probs, context),
        -8.199, 1e-2)


def test_ex_post_utility():
    context = get_context()

    occurred = context.decision_thresholds[10]
    econ_par = 0.1
    spend = 0.052
    assert np.isclose(
        _ex_post_utility(econ_par, occurred, spend, context),
        -3.386, 1e-2)

    occurred = context.decision_thresholds[10]
    econ_par = 0.7
    spend = 0.052
    assert np.isclose(
        _ex_post_utility(econ_par, occurred, spend, context),
        -3.847, 1e-2)

    occurred = context.decision_thresholds[10]
    econ_par = 0.1
    spend = 3
    assert np.isclose(
        _ex_post_utility(econ_par, occurred, spend, context),
        -8.199, 1e-2)


def test_find_spend_ensemble():
    context = get_context()

    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = _calc_likelihood(ens, context.decision_thresholds)
    econ_par = 0.1
    assert np.isclose(_find_spend_ensemble(econ_par, ens, probs, context), 0.012, 1e-1)

    # Not implemented to work with deterministic forecasts so no need to test for it.
    # Code uses analytical_spend method of economic model instead of find_spend for 
    # deterministic forecasts. Noting here because it could be a source of hard to 
    # find bugs in the future.


def test_single_timestep():
    t = 1
    ob = 10
    econ_par = 0.1

    context = get_context()

    np.random.seed(42)
    fcst = np.random.normal(10, 1, 100)
    ref = np.random.normal(5, 3, 100)

    t = 0
    result = single_timestep(t, econ_par, ob, fcst, ref, context)

    assert np.isclose(result['ob_spend'], 0.0076, 1e-2)
    assert np.isclose(result['fcst_spend'], 0.012, 1e-2)
    assert np.isclose(result['ref_spend'], 0.0076, 1e-2)


def test_calc_likelihoods():
    np.random.seed(42)

    # typical ensemble and range of thresholds
    ens = np.random.normal(10, 1, 100)
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(_calc_likelihood(ens, thresholds), np.array([0, 0, 0.01, 0.16, 0.37, 0.35, 0.11, 0, 0, 0]), 1e-1)

    # all in 1 class
    ens = np.random.normal(1000, 1, 100)
    thresholds = [0, 5]
    assert np.allclose(_calc_likelihood(ens, thresholds), np.array([0, 1]), 1e-1)

    # adds to 1
    assert np.equal(np.sum(_calc_likelihood(ens, thresholds)), 1)

    # Continuous decision with 100 member ensemble forecast
    assert np.array_equal(_calc_likelihood(ens, None), np.full(100, 1e-2))

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
    assert np.equal(_realised_threshold(0.5, thresholds), 0)
    assert np.equal(_realised_threshold(3, thresholds), 3)
    assert np.equal(_realised_threshold(3.5, thresholds), 3)
    assert np.equal(_realised_threshold(6, thresholds), 6)
    assert np.equal(_realised_threshold(7, thresholds), 6)

    with pytest.raises(ValueError):
        _realised_threshold(0.1, [1, 2, 3])

    with pytest.raises(ValueError):
        values = [0.5, 3, 3.5, 6, 7]
        _realised_threshold(values, thresholds)

    assert np.equal(_realised_threshold(42, None), 42)

    assert np.isnan(_realised_threshold(np.nan, thresholds))

