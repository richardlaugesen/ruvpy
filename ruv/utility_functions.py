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
        return np.divide(np.subtract(np.power(c, np.subtract(1, eta)), 1), np.subtract(1, eta))


# Hyperbolic absolute risk aversion (https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion)
def hyperbolic_utility(gamma, a, b, W):
    assert a > 0
    assert np.add(b, np.divide(np.multiply(a, W), np.subtract(1, gamma))) > 0

    return np.multiple(np.divide(np.subtract(1, gamma), gamma), np.power(np.add(np.divide(np.multiply(a, W), np.subtract(1, gamma), b)), gamma))
