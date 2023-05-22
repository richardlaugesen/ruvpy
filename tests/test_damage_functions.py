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

from ruv.damage_functions import *

def test_logistic():
    params = {'A': 1, 'k': 0.5, 'threshold': 0.5}
    damage_func = logistic(params)

    assert np.allclose(
        damage_func(np.array([0, 1, 2])),
        np.array([0.4378235, 0.5621765, 0.6791787]), 1e-5)


def test_logistic_zero():
    params = {'A': 1, 'k': 0.5, 'threshold': 0.5}
    damage_func = logistic_zero(params)

    assert np.allclose(
        damage_func(np.array([0, 1, 2])),
        np.array([0, 0.5621765, 0.6791787]), 1e-5)
    

def test_binary():
    params = {'max_loss': 2, 'min_loss': -1, 'threshold': 0.5}
    damage_func = binary(params)

    assert np.array_equal(
        damage_func(np.array([-1000, 0, 0.2, 0.5, 0.6, 1000])), 
        np.array([-1, -1, -1, 2, 2, 2]))


def test_linear():
    params = {'slope': 0, 'intercept': 0}
    damage_func = linear(params)

    assert np.array_equal(
        damage_func(np.array([-1000, 0, 0.2, 0.5, 0.6, 1000])), 
        np.array([0, 0, 0, 0, 0, 0]))
    
    params = {'slope': 5, 'intercept': 1}
    damage_func = linear(params)

    assert np.array_equal(
        damage_func(np.array([-1000, 0, 0.2, 0.5, 0.6, 1000])), 
        np.array([-4999, 1, 2, 3.5, 4, 5001]))    


def test_user_defined():

    points = [
        (0, 2000),
        (1, 0),
        (4, 0),
        (10, 10000)
    ]

    params = {'interpolator': user_defined_interpolator(points)}
    damage_func = user_defined(params)
    interpolator = user_defined_interpolator(points)

    values = np.random.rand(100) * 20000 - 1000
    assert np.array_equal(damage_func(values), interpolator(values))

    params = {'points': points}
    damage_func = user_defined(params)

    assert np.array_equal(damage_func(values), interpolator(values))


def test_user_defined_interpolator():

    points = [
        (0, 2000),
        (1, 0),
        (4, 0),
        (10, 10000)
    ]

    interpolator = user_defined_interpolator(points)

    assert np.allclose(
        interpolator(np.array([-1000, 0, 0.5, 1, 3, 4, 8, 10, 15, 1000])), 
        np.array([10000, 2000, 1000, 0, 0, 0, 6666, 10000, 10000, 10000]), 1e-1)