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

# Net expenses from cost-loss model
def cost_loss(values, spend, alpha, damage_function):
    damages = damage_function(values)
    benefits = np.minimum(np.divide(spend, alpha), damages)
    return np.subtract(np.subtract(benefits, damages), spend)


# Optimal cost-loss spend amount when forecast probability is entirely in a single flow class (ie. deterministic)
def cost_loss_analytical_spend(threshold, alpha, damage_function):
    return damage_function(threshold) * alpha
