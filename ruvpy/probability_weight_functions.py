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

# TODO: change from "weights" to "capactities"?

# Cumulative Prospect Theory probability weight function from Tversky & Kahneman (1992)
def power_weights(params: dict) -> Callable:
    exponent = params['exponent']

    def weight(p: np.ndarray) -> np.ndarray:
        return np.divide(np.power(p, exponent), np.power(np.add(np.power(p, exponent), np.power(np.subtract(1, p), exponent)), np.divide(1, exponent)))

    return weight


# Linear probability weight function
def linear_weights(params: dict) -> Callable:
    # Params not used

    def weight(p: np.ndarray) -> np.ndarray:
        return p

    return weight
