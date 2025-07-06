"""Damage functions used to translate event magnitude into monetary loss.

The functions implemented here cover a range of common shapes and can
be replaced with bespoke models if needed.  Each factory returns a
callable that maps event magnitude to damages.
"""

from typing import Callable, List, Tuple
import numpy as np
from scipy import interpolate


def logistic(params: dict) -> Callable:
    """Return a logistic damage function.

    Damages increase sigmoidally toward ``A`` with growth rate ``k``
    centred on ``threshold``. See
    https://en.wikipedia.org/wiki/Logistic_function for background.
    """
    A = params['A']
    k = params['k']
    threshold = params['threshold']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return A / (1 + np.exp(-k * (magnitude - threshold)))

    return damages


def logistic_zero(params: dict) -> Callable:
    """Logistic damage function pegged to zero for zero flow.

    Extends :func:`logistic` by forcing damages to ``0`` when the
    magnitude is ``0``. Useful where small events cause no loss.
    """
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
    """Return a binary damage function.

    Losses jump from ``min_loss`` to ``max_loss`` once ``threshold`` is
    exceeded, representing an on/off type impact.
    """
    threshold, max_loss, min_loss = params['threshold'], params['max_loss'], params['min_loss']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        ge = np.greater_equal(magnitude, threshold)
        damages = np.empty(magnitude.shape)
        damages[ge] = max_loss
        damages[~ge] = min_loss
        return damages

    return damages


def linear(params: dict) -> Callable:
    """Return a linear damage function.

    Damages vary linearly with magnitude using ``slope`` and
    ``intercept`` parameters.
    """
    slope, intercept = params['slope'], params['intercept']

    def damages(magnitude: np.ndarray) -> np.ndarray:
        return slope * magnitude + intercept

    return damages


def user_defined(params: dict) -> Callable:
    """Damage function interpolated over user-defined ``(x, y)`` points.

    Allows arbitrary curves by linearly interpolating between supplied
    points and extrapolating with the last value.
    """
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
