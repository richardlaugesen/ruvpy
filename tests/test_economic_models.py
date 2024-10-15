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

from ruv.economic_models import cost_loss, cost_loss_analytical_spend
from ruv.damage_functions import logistic_zero

def test_cost_loss_analytical_spend():

    params = {'A': 1, 'k': 0.5, 'threshold': 0.5}
    damage_func = logistic_zero(params)

    decision_threshold = 0.7; alpha = 0.2

    assert np.equal(
        cost_loss_analytical_spend(alpha, decision_threshold, damage_func), damage_func(decision_threshold) * alpha)


def test_cost_loss():

    params = {'A': 1, 'k': 0.5, 'threshold': 0.5}
    damage_func = logistic_zero(params)

    alpha = 0.1; value = 0.7; spend = 0
    assert np.isclose(cost_loss(alpha, value, spend, damage_func), -0.52, 1e-2)

    alpha = 0.1; value = 0.7; spend = 0.052
    assert np.isclose(cost_loss(alpha, value, spend, damage_func), -0.057, 1e-2)

    alpha = 0.1; value = 0.7; spend = 0.5
    assert np.isclose(cost_loss(alpha, value, spend, damage_func), -0.5, 1e-2)

    alpha = 0.6; value = 0.7; spend = 10
    assert np.isclose(cost_loss(alpha, value, spend, damage_func), -10, 1e-2)

    values = np.arange(0, 1, 0.1)
    spend = np.arange(0, 10, 1)

    alpha = 0.1
    assert np.allclose(
        cost_loss(alpha, values, spend, damage_func),
        np.array([0, -1, -2, -3, -4, -5, -6, -7, -8, -9]), 1e-1)

    alpha = 0.99
    assert np.allclose(
        cost_loss(alpha, values, spend, damage_func),
        np.array([0, -1, -2, -3, -4, -5, -6, -7, -8, -9]), 1e-1)
    