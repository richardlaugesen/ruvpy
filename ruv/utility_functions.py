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

# Constant Absolute Risk Aversion utility function
def cara(risk_aversion, outcome):
    return exponential_utility(risk_aversion, outcome)


# Constant Relaive Risk Aversion utility function
def crra(risk_aversion, outcome):
    return isoelastic_utility(risk_aversion, outcome)


# Exponential utility (https://en.wikipedia.org/wiki/Exponential_utility)
def exponential_utility(alpha, c):
    if alpha == 0:
        return c
    else:
        return np.divide(-np.exp(np.multiply(-alpha, c)), alpha)


# Isoelastic utility (https://en.wikipedia.org/wiki/Isoelastic_utility)
def isoelastic_utility(eta, c):
    if eta == 1:
        return np.log(c)
    else:        
        return np.divide(np.power(c, np.subtract(1, eta)), np.subtract(1, eta))


# Hyperbolic absolute risk aversion (https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion)
def hyperbolic_utility(g, a, b, W):
    if g == 0 or g == 1:
        raise Exception('g cannot be 0 or 1')

    if a <= 0:
        raise Exception('a > 0')
    
    if np.any(W < 0):
        raise Exception('W must be positive')

    if np.any(b + (a * W) / (1 - g) <= 0):
        raise Exception('b + (a * W) / (1 - g) > 0')

    return np.multiply(np.divide(np.subtract(1, g), g), np.power(np.add(np.divide(np.multiply(a, W), np.subtract(1, g)), b), g))

