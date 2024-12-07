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

from ruvpy.utility_functions import cara, crra, exponential_utility, isoelastic_utility, hyperbolic_utility


def test_cara():
    for risk_aversion in np.arange(0, 10, 0.5):
        cara_ut = cara({'A': risk_aversion})
        exp_ut = exponential_utility({'A': risk_aversion})

        for outcome in np.arange(0.1, 10, 0.5):
            assert np.equal(cara_ut(outcome), exp_ut(outcome))


def test_crra():
    for risk_aversion in np.arange(0, 10, 0.5):
        crra_ut = crra({'eta': risk_aversion})
        iso_ut = isoelastic_utility({'eta': risk_aversion})

        for outcome in np.arange(0.1, 10, 0.5):
            assert np.equal(crra_ut(outcome), iso_ut(outcome))


def test_exponential_utility():
    with pytest.raises(KeyError):
        exponential_utility({'B': 0.1})

    assert np.isclose(exponential_utility({'A': 0.1})(10), -3.7, rtol=1e-2, atol=1e-3)
    assert np.isclose(exponential_utility({'A': 0})(10), 10, rtol=1e-2, atol=1e-3)
    assert np.isclose(exponential_utility({'A': -0.1})(10), 27.2, rtol=1e-2, atol=1e-3)
    assert np.isclose(exponential_utility({'A': 50})(1000), 0, rtol=1e-2, atol=1e-3)
    assert np.isclose(exponential_utility({'A': 0.1})(100000), 0, rtol=1e-2, atol=1e-3)

    outcomes = np.array([-10, 0, 0.5, 10, 1000, 1e10])

    risk_aversion = 0
    assert np.allclose(
        exponential_utility({'A': risk_aversion})(outcomes),
        outcomes)

    risk_aversion = 0.1
    assert np.allclose(
        exponential_utility({'A': risk_aversion})(outcomes),
        np.array([-27.2, -10, -9.5, -3.7, 0, 0]), rtol=1e-2, atol=1e-3)

    risk_aversion = 5
    assert np.allclose(
        exponential_utility({'A': risk_aversion})(outcomes),
        np.array([-1.04e+21, -0.2, -0.017, 0, 0, 0]), rtol=1e-2, atol=1e-3)


def test_isoelastic_utility():
    with pytest.raises(KeyError):
        isoelastic_utility({'B': 0.1})

    with pytest.raises(ValueError):
        isoelastic_utility({'eta': 0.1})(0)

    assert np.isclose(isoelastic_utility({'eta': 0.1})(10), 7.714758163603128, rtol=1e-2, atol=1e-3)
    assert np.isclose(isoelastic_utility({'eta': 0})(10), 9, rtol=1e-2, atol=1e-3)
    assert np.isclose(isoelastic_utility({'eta': 1})(10), np.log(10), rtol=1e-2, atol=1e-3)
    assert np.isclose(isoelastic_utility({'eta': 1.5})(10), (np.power(10, -0.5) - 1) / -0.5, rtol=1e-2, atol=1e-3)
    assert np.isclose(isoelastic_utility({'eta': -0.1})(10), 10.535685561765158, rtol=1e-2, atol=1e-3)
    assert np.isclose(isoelastic_utility({'eta': 50.0})(1000.0), 0.02040816326530612, rtol=1e-2, atol=1e-3)
    assert np.isclose(isoelastic_utility({'eta': 0.1})(100000), 35135.30733520423, rtol=1e-2, atol=1e-3)

    assert np.isnan(isoelastic_utility({'eta': 0.1})(-10))

    assert np.isclose(
        isoelastic_utility({'eta': 0.1, 'symmetric': True})(-10),
        -isoelastic_utility({'eta': 0.1})(10), rtol=1e-2, atol=1e-3)

    outcomes = np.array([0.5, 10, 1000])

    risk_aversion = 0
    assert np.allclose(
        isoelastic_utility({'eta': risk_aversion})(outcomes),
        outcomes - 1, rtol=1e-2, atol=1e-3)

    risk_aversion = 1
    assert np.allclose(
        isoelastic_utility({'eta': risk_aversion})(outcomes),
        np.log(outcomes), rtol=1e-2, atol=1e-3)

    risk_aversion = 0.1
    assert np.allclose(
        isoelastic_utility({'eta': risk_aversion})(outcomes),
        np.array([-0.51568, 7.714758, 555.76359]), rtol=1e-2, atol=1e-3)

    risk_aversion = 5
    assert np.allclose(
        isoelastic_utility({'eta': risk_aversion})(outcomes),
        np.array([-3.75, 0.249975, 0.25]), rtol=1e-2, atol=1e-3)


def test_hyperbolic_utility():
    with pytest.raises(KeyError):
        hyperbolic_utility({'g': 0.1, 'a': 0.1, 'Q': 0.1})

    with pytest.raises(ValueError):
        hyperbolic_utility({'g': 1, 'a': 1, 'b': 0})(10)

    with pytest.raises(ValueError):
        hyperbolic_utility({'g': 0.7, 'a': -1, 'b': 0})(10)

    with pytest.raises(ValueError):
        hyperbolic_utility({'g': 0.7, 'a': 1, 'b': 0})(0)

    with pytest.raises(ValueError):
        hyperbolic_utility({'g': 0.7, 'a': 0.2, 'b': 1})(-20)

    assert np.isclose(
        hyperbolic_utility({'g': 0.7, 'a': 0.2, 'b': 1})(0),
        0.428571, rtol=1e-2, atol=1e-3)

    outcomes = np.array([0.5, 10, 1000])

    g = 1e10
    a = 0.2
    b = 1
    w = outcomes
    assert np.allclose(
        hyperbolic_utility({'g': g, 'a': a, 'b': b})(w),
        exponential_utility({'A': a})(w) * a, rtol=1e-2, atol=1e-3)

    g = 1e-7
    a = 1
    b = 0
    w = outcomes
    assert np.allclose(
        hyperbolic_utility({'g': g, 'a': a, 'b': b})(w),
        np.log(w) + a/g, rtol=1e-2, atol=1e-3)

    g = 1e-7
    a = 1
    b = 1
    w = outcomes
    assert np.allclose(
        hyperbolic_utility({'g': g, 'a': a, 'b': b})(w),
        np.log(a * w + b) + a/g, rtol=1e-2, atol=1e-3)

    hyperbolic_utility({'g': 0.7, 'a': 1, 'b': 1})(0)

    g = 0.7
    a = 1 - g
    b = 0
    e = 1 - g
    w = outcomes
    hyp = hyperbolic_utility({'g': g, 'a': a, 'b': b})(w)
    iso = isoelastic_utility({'eta': e})(w) * e + e/(1-e)  # correction factor needed since we have -1 in numerator
    assert np.allclose(
        hyp, iso, rtol=1e-2, atol=1e-3)
