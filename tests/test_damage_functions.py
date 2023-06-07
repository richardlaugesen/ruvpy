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