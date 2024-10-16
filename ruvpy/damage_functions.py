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

from typing import Callable, List, Tuple
import numpy as np
from scipy import interpolate


def logistic(params: dict) -> Callable:
    A = params['A']
    k = params['k']
    threshold = params['threshold']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return A / (1 + np.exp(-k * (magnitude - threshold)))

    return damages


# Same as logistic with damages pegged to zero for zero flow
def logistic_zero(params: dict) -> Callable:
    logistic_closure = logistic(params)

    def damages(magnitude: np.ndarray) -> np.ndarray:
        damages = logistic_closure(magnitude)
        try:
            damages[magnitude == 0] = 0
        except TypeError:
            damages = 0 if magnitude == 0 else damages
        return damages

    return damages


def binary(params: dict) -> Callable:
    threshold, max_loss, min_loss = params['threshold'], params['max_loss'], params['min_loss']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        ge = np.greater_equal(magnitude, threshold)
        damages = np.empty(magnitude.shape)
        damages[ge] = max_loss
        damages[~ge] = min_loss
        return damages

    return damages


def linear(params: dict) -> Callable:
    slope, intercept = params['slope'], params['intercept']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return slope * magnitude + intercept

    return damages


# Damages by a linear interpolation over a set of points (list of tuples)
def user_defined(params: dict) -> Callable:
    if 'interpolator' in params:
        inter = params['interpolator']
    else:
        points = params['points']
        inter = _user_defined_interpolator(points)

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return inter(magnitude)

    return damages


def _user_defined_interpolator(points: List[Tuple]) -> Callable:
    user_flows, user_damages = zip(*points)
    extrapolate_value = user_damages[-1]
    inter = interpolate.interp1d(
        user_flows, user_damages, kind='linear', fill_value=extrapolate_value, bounds_error=False)
    return inter
