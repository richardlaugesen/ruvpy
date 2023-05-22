# Copyright 2023 Richard Laugesen
#
# This file is part of RUV
#
# RUV is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RUV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RUV.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# Net expenses from cost-loss model
def cost_loss(values, spend, alpha, damage_function):
    damages = damage_function(values)
    benefits = np.minimum(np.divide(spend, alpha), damages)
    return np.subtract(np.subtract(benefits, damages), spend)


# Optimal cost-loss spend amount when forecast probability is entirely in a single flow class (ie. deterministic)
def cost_loss_analytical_spend(threshold, alpha, damage_function):
    return damage_function(threshold) * alpha
