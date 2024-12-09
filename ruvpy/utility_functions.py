

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

from typing import Callable, Union
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

    def utility(c: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if A == 0:
            return c
        else:
            # using expm1 to reduce chance of overflows
            return (-1 - np.expm1(-A * c)) / A

    return utility


# Isoelastic utility function (https://en.wikipedia.org/wiki/Isoelastic_utility)
def isoelastic_utility(params: dict):
    eta = float(params['eta'])
    symmetric = params.get('symmetric', False)

    def utility(c: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        c = _ensure_float(c)

        if np.any(c == 0) and eta > 1:
            raise ValueError("Amount cannot be 0 for isoelastic utility when $\eta$ > 1")

        # For symmetric CRRA behaviour with negative values
        val = np.abs(c) if symmetric else c

        if eta == 1:
            result = np.log(val)
        else:
            result = (np.power(val, 1 - eta) - 1) / (1 - eta)

        # For symmetric CRRA behaviour with negative values
        if symmetric:
            result *= np.sign(c)

        return result

    return utility


# Hyperbolic absolute risk aversion utility function (https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion)
def hyperbolic_utility(params: dict) -> Callable:
    g, a, b = params['g'], params['a'], params['b']

    if a <= 0:
        raise ValueError("Parameter 'a' must be > 0")
    if g == 0:
        raise ValueError("Parameter 'g' cannot be 0")
    if g == 1:
        raise ValueError("Parameter 'g' cannot be 1")

    def utility(W: Union[np.ndarray, float]) -> Union[np.ndarray, float]:

        W = _ensure_float(W)

        if np.any(b + (a * W) / (1 - g) <= 0):
            raise ValueError('b + (a * W) / (1 - g) must be > 0')

        return ((1 - g) / g) * np.power((a * W / (1 - g) + b), g)

    return utility


# Cumulative Prospect Theory value function from Tversky & Kahneman (1992)
def power_value(params: dict) -> callable:
    alpha, beta, lambda_ = params['alpha'], params['beta'], params['lambda']

    def value(c: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        c = np.asarray(c)
        result = np.empty_like(c, dtype=float)

        mask = c >= 0

        result[mask] = np.power(c[mask], alpha)
        result[~mask] = -lambda_ * np.power(-c[~mask], beta)

        return result if result.size > 1 else result.item()

    return value


def _ensure_float(input_data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return input_data.astype(float) if isinstance(input_data, np.ndarray) else float(input_data)
