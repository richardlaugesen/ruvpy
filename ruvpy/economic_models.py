# Copyright 2021–2024 Richard Laugesen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Economic models representing the cost and benefit structure.

These functions provide simple formulations for converting spending
and damages into a net utility.
"""

import numpy as np


def cost_loss(alpha: float, values: np.ndarray, spend: float, damage_function: callable) -> np.ndarray:
    """Standard cost-loss model.

    Net utility is calculated as benefits minus damages and spending,
    with benefits capped at ``spend / alpha``.
    """
    damages = damage_function(values)
    benefits = np.minimum(spend/alpha, damages)
    return benefits - damages - spend


def cost_loss_analytical_spend(alpha: float, values: float, damage_function: callable) -> float:
    """Analytical solution for the optimal spend in a cost-loss model.

    Computes the spend that maximises expected profit for a
    deterministic forecast.
    """
    return damage_function(values) * alpha
