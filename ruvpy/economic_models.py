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
"""Economic models representing the net outcome of damages and user spending.

These functions provide a net economic outcome from user spending ``spend`` and
the damages due to the state of the world ``values`` and the ``damage_function``.
An economic model will typically be parameterised (single parameter only for
current RUVPY version).

An additional function can be provided for each economic model to calculate
the optimal spend amount for deterministic inputs. This significantly speeds
up the calculation of RUV by avoiding the numerical optimisation required
with ensemble inputs.
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
    deterministic input.
    """
    return damage_function(values) * alpha
