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

# TODO: make these work with a single float as well as an array

# Cumulative Prospect Theory probability weight function from Tversky & Kahneman (1992)
def power_weights(params: dict) -> Callable:
    exponent = params['exponent']

    def weight(p: np.ndarray) -> np.ndarray:
        w = np.zeros_like(p)

        # values between 0 and 1
        mask = (p > 0) & (p < 1)
        p_masked = p[mask]
        numerator = np.power(p_masked, exponent)
        denominator = np.power(
            np.power(p_masked, exponent) + np.power(1 - p_masked, exponent),
            1 / exponent
        )
        w[mask] = numerator / denominator

        # values <=0 and >=1
        w[p <= 0] = 0
        w[p >= 1] = 1

        return w

    return weight


# Linear probability weight function
def linear_weights(params: dict) -> Callable:
    # Params not used

    def weight(p: np.ndarray) -> np.ndarray:
        return p

    return weight
