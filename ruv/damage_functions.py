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

from typing import Callable, Dict, List, Tuple, Union
import numpy as np
from scipy import interpolate

#


def logistic(params: Dict[str, float]) -> Callable[[float], float]:
    """
    Calculates logistic-based damages.

    Parameters:
        params (Dict[str, float]): A dictionary containing the parameters A, k, and threshold.

    Returns:
        Callable[[float], float]: A function that takes a magnitude as input and returns the damages.

    Notes:
        Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.            
    """
    A = params['A']
    k = params['k']
    threshold = params['threshold']

    def damages(magnitude: float) -> float:
        """
        Calculates the damages based on the given magnitude.

        Parameters:
            magnitude (float): The magnitude.

        Returns:
            float: The damages.

        Notes:
            Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.            
        """
        return np.divide(A, np.add(1, np.exp(np.multiply(-k, np.subtract(magnitude, threshold)))))

    return damages


def logistic_zero(params: Dict[str, Union[float, np.ndarray]]) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Logistic-based damages but with damages=0 when magnitude=0.

    Args:
        params: A dictionary of parameters.

    Returns:
        A function that calculates damages based on the logistic function.

    Notes:
        Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.            
    """
    logistic_curry = logistic(params)

    def damages(magnitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        damages = logistic_curry(magnitude)
        try:
            damages[magnitude == 0] = 0
        except TypeError:
            damages = 0 if magnitude == 0 else damages
        return damages

    return damages


def binary(params: Dict[str, float]) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Binary damages, 'max_loss' above a 'threshold', 'min_loss' below

    Args:
        params: A dictionary of parameters containing threshold, max_loss, and min_loss

    Returns:
        A function that calculates binary damages.

    Notes:
        Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.                    
    """
    threshold, max_loss, min_loss = params['threshold'], params['max_loss'], params['min_loss']

    def damages(magnitude: Union[float, np.ndarray]) -> np.ndarray:
        ge = np.greater_equal(magnitude, threshold)
        damages = np.empty(magnitude.shape)
        damages[ge] = max_loss
        damages[~ge] = min_loss
        return damages

    return damages


def linear(params: Dict[str, float]) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Linear damages.

    Args:
        params: A dictionary of parameters.

    Returns:
        A function that calculates linear damages.

    Notes:
        Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.                    
    """
    slope, intercept = params['slope'], params['intercept']

    def damages(magnitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return slope * magnitude + intercept

    return damages


def user_defined(params: Dict[str, Union[List[Tuple[float, float]], Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]]]) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    User defined damages by a linear interpolation over a set of points (list of tuples).

    Args:
        params: A dictionary of parameters.

    Returns:
        A function that calculates user-defined damages.

    Notes:
        Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.                    
    """
    if 'interpolator' in params:
        inter = params['interpolator']
    else:
        points = params['points']
        inter = user_defined_interpolator(points)

    def damages(magnitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return inter(magnitude)

    return damages


def user_defined_interpolator(points: List[Tuple[float, float]]) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Convenience function for investigating interpolator used by user_defined.

    Args:
        points: A list of tuples containing flow and damage values.

    Returns:
        An interpolating function.

    Notes:
        Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.                    
    """
    user_flows, user_damages = zip(*points)
    extrapolate_value = user_damages[-1]
    inter = interpolate.interp1d(
        user_flows, user_damages, kind='linear', fill_value=extrapolate_value, bounds_error=False)
    return inter
