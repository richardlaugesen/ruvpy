# Copyright 2020–2023 Richard Laugesen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utility functions describing decision-maker preferences.

Each factory returns a callable that converts monetary outcomes to
utility according to a particular risk preference model.
"""

from typing import Callable
import numpy as np


def cara(params: dict) -> Callable:
    """Constant Absolute Risk Aversion (CARA).

    Returns an exponential utility function with risk-aversion
    coefficient ``A``. Absolute risk aversion remains constant
    regardless of wealth. See
    https://en.wikipedia.org/wiki/Risk_aversion#Constant_absolute_risk_aversion.
    """
    return exponential_utility(params)


def crra(params: dict) -> Callable:
    """Constant Relative Risk Aversion (CRRA).

    Wraps :func:`isoelastic_utility` so relative risk aversion ``eta``
    remains constant regardless of wealth.
    """
    return isoelastic_utility(params)


def exponential_utility(params: dict) -> Callable:
    """Exponential utility function implementing CARA behaviour.

    This formulation is common in economics and underpins constant
    absolute risk aversion.
    """
    A = params['A']

    def utility(c: float) -> float:
        if A == 0:
            return c
        else:
            # using expm1 to reduce chance of overflows
            return (-1 - np.expm1(-A * c)) / A

    return utility


def isoelastic_utility(params: dict) -> Callable:
    """Isoelastic (power) utility function for CRRA behaviour.

    Parameter ``eta`` controls relative risk aversion. ``eta=1`` yields
    logarithmic utility.
    """
    eta = float(params['eta'])

    def utility(c: float) -> float:
        c = _ensure_float(c)

        if eta == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - eta) / (1 - eta)

    return utility


def hyperbolic_utility(params: dict) -> Callable:
    """Hyperbolic Absolute Risk Aversion (HARA) utility function.

    Generalises both CARA and CRRA through parameters ``g``, ``a`` and
    ``b``.
    """
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
    """Cast ``input_data`` to ``float`` regardless of input type."""
    return input_data.astype(float) if isinstance(input_data, np.ndarray) else float(input_data)
