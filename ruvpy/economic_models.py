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

import numpy as np


def cost_loss(alpha: float, values: np.ndarray, spend: float, damage_function: callable) -> np.ndarray:
    damages = damage_function(values)
    benefits = np.minimum(spend/alpha, damages)
    return benefits - damages - spend


def cost_loss_analytical_spend(alpha: float, values: float, damage_function: callable) -> float:
    return damage_function(values) * alpha
