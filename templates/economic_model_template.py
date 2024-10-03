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
# ECONOMIC MODEL
#
# Note for now it does not use the closure
# pattern as in the other components and only
# accepts single parameter economic models
# ------------------------------------------------

def economic_model_template(econ_param_1: float, values: np.ndarray, spend: float, damage_function: Callable) -> np.ndarray:
    damages = damage_function(values)
    outcome = damages * econ_param_1 - spend
    return outcome


# fast analytical solution for spend, or return None
def economic_model_analytical_spend_template(econ_param_1: float, value: float, damage_function: callable) -> float:
    return None

