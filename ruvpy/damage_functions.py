from typing import Callable, List, Tuple
import numpy as np
from scipy import interpolate


def logistic(params: dict) -> Callable:
    """Return a logistic damage function."""
    A = params['A']
    k = params['k']
    threshold = params['threshold']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return A / (1 + np.exp(-k * (magnitude - threshold)))

    return damages


def logistic_zero(params: dict) -> Callable:
    """Return the logistic damage function pegged to zero for zero flow."""
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
    """Return a binary damage function."""
    threshold, max_loss, min_loss = params['threshold'], params['max_loss'], params['min_loss']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        ge = np.greater_equal(magnitude, threshold)
        damages = np.empty(magnitude.shape)
        damages[ge] = max_loss
        damages[~ge] = min_loss
        return damages

    return damages


def linear(params: dict) -> Callable:
    """Return a linear damage function."""
    slope, intercept = params['slope'], params['intercept']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return slope * magnitude + intercept

    return damages


def user_defined(params: dict) -> Callable:
    """Return a damage function interpolated over user-defined ``(x, y)`` points."""
    if 'interpolator' in params:
        inter = params['interpolator']
    else:
        points = params['points']
        inter = _user_defined_interpolator(points)

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return inter(magnitude)

    return damages


def _user_defined_interpolator(points: List[Tuple]) -> Callable:
    """Create an interpolation function from a list of ``(x, y)`` points."""
    user_flows, user_damages = zip(*points)
    extrapolate_value = user_damages[-1]
    inter = interpolate.interp1d(
        user_flows, user_damages, kind='linear', fill_value=extrapolate_value, bounds_error=False)
    return inter
