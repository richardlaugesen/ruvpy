from typing import Callable
import numpy as np


def cara(params: dict) -> Callable:
    """Constant Absolute Risk Aversion utility function."""
    return exponential_utility(params)


def crra(params: dict) -> Callable:
    """Constant Relative Risk Aversion utility function."""
    return isoelastic_utility(params)


def exponential_utility(params: dict) -> Callable:
    """Exponential utility function used for CARA behaviour.

    See https://en.wikipedia.org/wiki/Exponential_utility for details.
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
    """Isoelastic utility function used for CRRA behaviour.

    See https://en.wikipedia.org/wiki/Isoelastic_utility.
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
    """Hyperbolic Absolute Risk Aversion utility function.

    See https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion.
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
