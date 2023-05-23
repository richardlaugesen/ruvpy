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

from ruv.utility_functions import *
import pytest

def test_cara():
    for risk_aversion in np.arange(0, 10, 0.5):
        for outcome in np.arange(0, 10, 0.5):
            print(risk_aversion, outcome)
            assert np.equal(cara(risk_aversion, outcome), exponential_utility(risk_aversion, outcome))


def test_crra():
    for risk_aversion in np.arange(0, 10, 0.5):
        for outcome in np.arange(0, 10, 0.5):
            print(risk_aversion, outcome)
            assert np.equal(crra(risk_aversion, outcome), isoelastic_utility(risk_aversion, outcome))


def test_exponential_utility():

    assert np.isclose(exponential_utility(0.1, 10), -3.7, 1e-1)
    assert np.isclose(exponential_utility(0, 10), 10, 1e-1)
    assert np.isclose(exponential_utility(-0.1, 10), 27.2, 1e-1)
    assert np.isclose(exponential_utility(50, 1000), 0, 1e-1)
    assert np.isclose(exponential_utility(0.1, 100000), 0, 1e-1)

    outcomes = np.array([-10, 0.5, 0, 0.5, 10, 1000])

    risk_aversion = 0
    assert np.allclose(
        exponential_utility(risk_aversion, outcomes),
        outcomes)
    
    risk_aversion = 0.1
    assert np.allclose(
        exponential_utility(risk_aversion, outcomes),
        np.array([-27.2, -9.5, -10, -9.5, -3.7, 0]), 1e-1)
    
    risk_aversion = 5
    assert np.allclose(
        exponential_utility(risk_aversion, outcomes),
        np.array([-1.04e+21, -0.017, -0.2, -0.017, 0, 0]), 1e-1)


def test_isoelastic_utility():

    assert np.isclose(isoelastic_utility(0.1, 10), 8.8, 1e-1)
    assert np.isclose(isoelastic_utility(0, 10), 10, 1e-1)
    assert np.isclose(isoelastic_utility(1, 10), np.log(10), 1e-1)
    assert np.isclose(isoelastic_utility(1.5, 10), np.power(10, -0.5) / -0.5, 1e-1)
    assert np.isclose(isoelastic_utility(-0.1, 10), 11.4, 1e-1)
    assert np.isclose(isoelastic_utility(50.0, 1000.0), 0, 1e-1)
    assert np.isclose(isoelastic_utility(0.1, 100000), 35136.4, 1e-1)
    assert np.isnan(isoelastic_utility(0.1, -10))

    outcomes = np.array([0.5, 0, 0.5, 10, 1000])

    risk_aversion = 0
    assert np.allclose(
        isoelastic_utility(risk_aversion, outcomes),
        outcomes)
    
    risk_aversion = 1
    assert np.allclose(
        isoelastic_utility(risk_aversion, outcomes),
        np.log(outcomes))    
    
    risk_aversion = 0.1
    assert np.allclose(
        isoelastic_utility(risk_aversion, outcomes),
        np.array([0.6, 0, 0.6, 8.8, 556.9]), 1e-1)
    
    risk_aversion = 5
    assert np.allclose(
        isoelastic_utility(risk_aversion, outcomes),
        np.array([-4, -np.inf, -4, -0.000025, 0]), 1e-1)
    

def test_hyperbolic_utility():
    with pytest.raises(Exception):
        hyperbolic_utility(0, 1, 0, 10)

    with pytest.raises(Exception):
        hyperbolic_utility(1, 1, 0, 10)

    with pytest.raises(Exception):
        hyperbolic_utility(0.7, -1, 0, 10)

    with pytest.raises(Exception):
        hyperbolic_utility(0.7, 1, 0, 0)

    with pytest.raises(Exception):
        hyperbolic_utility(0.7, 0.2, 1, -20)

    g = 0.9999999; a = 1; b = 1; W = 10
    assert np.isclose(hyperbolic_utility(g, a, b, W), W, 1e-1)

    g = -1e10; a = 0.2; b = 1; W = 16
    assert np.isclose(hyperbolic_utility(g, a, b, W), exponential_utility(a, W) * a, 1e-1)

    g = 0.1; a = 1 - g; b = 1; W = 16  # W=0 fails
    assert np.isclose(hyperbolic_utility(g, a, b, W), isoelastic_utility(a, W) * a, 1e-1)

    g = 1e-7; a = 1; b = 0; W = 16
    assert np.isclose(hyperbolic_utility(g, a, b, W), np.log(W) + a/g, 1e-1)

    g = 1e-7; a = 1; b = 1; W = 16
    assert np.isclose(hyperbolic_utility(g, a, b, W), np.log(a * W + b) + a/g, 1e-1)

    outcomes = np.array([0.5, 20, 0.5, 10, 1000])
    g = 1e-5; a = 1 - g; b = 1
    assert np.allclose(
        hyperbolic_utility(g, a, b, outcomes),
        isoelastic_utility(a, outcomes) * a, 1e-1)
