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

from typing import Callable
import numpy as np


# Constant Absolute Risk Aversion utility function
def cara(params: dict) -> Callable:
    return exponential_utility(params)


# Constant Relative Risk Aversion utility function
def crra(params: dict) -> Callable:
    return isoelastic_utility(params)


# Exponential utility function (https://en.wikipedia.org/wiki/Exponential_utility)
def exponential_utility(params: dict) -> Callable:
    A = params['A']

    def utility(c: float) -> float:
        if A == 0:
            return c
        else:
            # using expm1 to reduce chance of overflows
            return (-1 - np.expm1(-A * c)) / A

    return utility


# Isoelastic utility function (https://en.wikipedia.org/wiki/Isoelastic_utility)
def isoelastic_utility(params: dict) -> Callable:
    eta = float(params['eta'])

    def utility(c: float) -> float:
        c = _ensure_float(c)

        if eta == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - eta) / (1 - eta)

    return utility


# Hyperbolic absolute risk aversion utility function (https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion)
def hyperbolic_utility(params: dict) -> Callable:
    g, a, b = params['g'], params['a'], params['b']

    def utility(W: float) -> float:
        if g == 0 or g == 1:
            raise Exception('g cannot be 0 or 1')

        if a <= 0:
            raise Exception('a > 0')

        if np.any(W < 0):
            raise Exception('W must be positive')

        if np.any(b + (a * W) / (1 - g) <= 0):
            raise Exception('b + (a * W) / (1 - g) > 0')

        return ((1 - g) / g) * np.power(((a * W) / (1 - g) + b), g)

    return utility


def _ensure_float(input_data: np.ndarray) -> float:
    return input_data.astype(float) if isinstance(input_data, np.ndarray) else float(input_data)
