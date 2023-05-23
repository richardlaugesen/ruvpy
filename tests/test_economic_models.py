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

from ruv.economic_models import *
from ruv.damage_functions import logistic_zero

def test_cost_loss_analytical_spend():

    params = {'A': 1, 'k': 0.5, 'threshold': 0.5}
    damage_func = logistic_zero(params)

    decision_threshold = 0.7; alpha = 0.2

    assert np.equal(
        cost_loss_analytical_spend(decision_threshold, alpha, damage_func),
        damage_func(decision_threshold) * alpha)


def test_cost_loss():

    params = {'A': 1, 'k': 0.5, 'threshold': 0.5}
    damage_func = logistic_zero(params)

    alpha = 0.1; value = 0.7; spend = 0
    assert np.isclose(cost_loss(value, spend, alpha, damage_func), -0.52, 1e-2)

    alpha = 0.1; value = 0.7; spend = 0.052
    assert np.isclose(cost_loss(value, spend, alpha, damage_func), -0.057, 1e-2)

    alpha = 0.1; value = 0.7; spend = 0.5
    assert np.isclose(cost_loss(value, spend, alpha, damage_func), -0.5, 1e-2)

    alpha = 0.6; value = 0.7; spend = 10
    assert np.isclose(cost_loss(value, spend, alpha, damage_func), -10, 1e-2)

    values = np.arange(0, 1, 0.1)
    spend = np.arange(0, 10, 1)

    alpha = 0.1
    assert np.allclose(
        cost_loss(values, spend, alpha, damage_func),
        np.array([0, -1, -2, -3, -4, -5, -6, -7, -8, -9]), 1e-1)

    alpha = 0.99
    assert np.allclose(
        cost_loss(values, spend, alpha, damage_func),
        np.array([0, -1, -2, -3, -4, -5, -6, -7, -8, -9]), 1e-1)
    