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

from typing import Dict, Callable, Union
import numpy as np


def cara(params: Dict[str, Union[int, float]]) -> Callable[[float], float]:
    """
    Constant Absolute Risk Aversion utility function.

    Parameters:
    - params: A dictionary containing the absolute risk aversion parameter 'A' for the utility function.

    Returns:
    - utility: A function that calculates the utility given a consumption value 'c'.

    Notes:
    - Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.    
    """
    return exponential_utility(params)


def crra(params: Dict[str, Union[int, float]]) -> Callable[[float], float]:
    """
    Constant Relative Risk Aversion utility function.

    Parameters:
    - params: A dictionary containing the relative risk aversion 'eta' for the utility function.

    Returns:
    - utility: A function that calculates the utility given a consumption value 'c'.

    Notes:
    - Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.    
    """
    return isoelastic_utility(params)


def exponential_utility(params: Dict[str, Union[int, float]]) -> Callable[[float], float]:
    """
    Exponential utility function (https://en.wikipedia.org/wiki/Exponential_utility).

    Parameters:
    - params: A dictionary containing the absolute risk aversion parameter 'A' for the utility function.

    Returns:
    - utility: A function that calculates the utility given a consumption value 'c'.

    Notes:
    - Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.    
    """
    A = params['A']

    def utility(c: float) -> float:
        if A == 0:
            return c
        else:
            # using expm1 to reduce chance of overflows
            return np.divide(np.subtract(-1, np.expm1(np.multiply(-A, c))), A)

    return utility


def ensure_float(input_data: Union[float, np.ndarray]) -> float:
    """
    Ensures that the input data is of type float.

    Parameters:
    - input_data: The input data.

    Returns:
    - The input data as a float.
    """
    return input_data.astype(float) if isinstance(input_data, np.ndarray) else float(input_data)


def isoelastic_utility(params: Dict[str, Union[int, float]]) -> Callable[[float], float]:
    """
    Isoelastic utility function (https://en.wikipedia.org/wiki/Isoelastic_utility).

    Parameters:
    - params: A dictionary containing the relative risk aversion parameter 'eta' for the utility function.

    Returns:
    - utility: A function that calculates the utility given a consumption value 'c'.

    Notes:
    - Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.    
    """
    eta = float(params['eta'])

    def utility(c: float) -> float:
        c = ensure_float(c)

        if eta == 1:
            return np.log(c)
        else:
            return np.divide(np.power(c, np.subtract(1, eta)), np.subtract(1, eta))

    return utility


def hyperbolic_utility(params: Dict[str, Union[int, float]]) -> Callable[[float], float]:
    """
    Hyperbolic absolute risk aversion utility function (https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion).

    Parameters:
    - params: A dictionary containing the parameters 'g', 'a', and 'b' for the utility function.

    Returns:
    - utility: A function that calculates the utility given a wealth value 'W'.

    Notes:
    - Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV.
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

        return np.multiply(np.divide(np.subtract(1, g), g), np.power(np.add(np.divide(np.multiply(a, W), np.subtract(1, g)), b), g))

    return utility
