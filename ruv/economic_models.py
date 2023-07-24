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

import numpy as np


def cost_loss(alpha: float, values: np.ndarray, spend: float, damage_function: callable) -> np.ndarray:
    """
    Calculates the net expenses using the cost-loss model.

    Parameters:
    alpha (float): The alpha value.
    values (np.ndarray): An array of values.
    spend (float): The spend amount.
    damage_function (callable): A function that calculates the damages.

    Returns:
    np.ndarray: An array of net expenses.
    """
    damages = damage_function(values)
    benefits = np.minimum(np.divide(spend, alpha), damages)
    return np.subtract(np.subtract(benefits, damages), spend)


def cost_loss_analytical_spend(alpha: float, threshold: float, damage_function: callable) -> float:
    """
    Calculates the optimal cost-loss spend amount when forecast probability is entirely in a single flow class.

    Parameters:
    alpha (float): The alpha value.
    threshold (float): The threshold value.
    damage_function (callable): A function that calculates the damages.

    Returns:
    float: The optimal cost-loss spend amount.
    """
    return damage_function(threshold) * alpha
