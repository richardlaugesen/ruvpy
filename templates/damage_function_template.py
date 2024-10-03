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

from typing import Callable
import numpy as np


# ------------------------------------------------
# DAMAGE FUNCTION
# ------------------------------------------------

def damage_function_template(params: dict) -> Callable:

    # extract parameters from the dictionary if needed
    damage_function_param_1 = params['damage_function_param_1']
    damage_function_param_2 = params['damage_function_param_2']
    # ...

    # define the actual damage function
    def damage_function(magnitude: np.ndarray) -> np.ndarray:

        # do something with the parameters, perhaps adjust the magnitude
        magnitude = magnitude / damage_function_param_1

        return magnitude

    return damage_function
