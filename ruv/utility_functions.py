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
from scipy.optimize import root_scalar

# Functional currying is used to allow a pre-paramterised utility functions to be passed into RUV

# Constant Absolute Risk Aversion utility function
def cara(params):
    return exponential_utility(params)


# Constant Relaive Risk Aversion utility function
def crra(params):
    return isoelastic_utility(params)


# Exponential utility (https://en.wikipedia.org/wiki/Exponential_utility)
def exponential_utility(params):
    A = params['A']

    def utility(c):
        if A == 0:
            return c
        else:
            return np.divide(-np.exp(np.multiply(-A, c)), A)

    return utility


# Isoelastic utility (https://en.wikipedia.org/wiki/Isoelastic_utility)
def isoelastic_utility(params):
    eta = params['eta']

    def utility(c):    
        if eta == 1:
            return np.log(c)
        else:        
            return np.divide(np.power(c, np.subtract(1, eta)), np.subtract(1, eta))

    return utility


# Hyperbolic absolute risk aversion (https://en.wikipedia.org/wiki/Hyperbolic_absolute_risk_aversion)
def hyperbolic_utility(params):
    g, a, b = params['g'], params['a'], params['b']

    def utility(W):        
        if g == 0 or g == 1:
            raise Exception('g cannot be 0 or 1')

        if a <= 0:
            raise Exception('a > 0')
        
        if np.any(W < 0):
            raise Exception('W must be positive')

        if np.any(b + (a * W) / (1 - g) <= 0):
            raise Exception('b + (a * W) / (1 - g) > 0')

        return np.multiply(np.divide(np.subtract(1, g), g), np.power(np.add(np.divide(np.multiply(a, W), np.subtract(1, g)), b), g))

    return utility


# Calculate CARA risk premium coefficient from risk aversion coefficient and gamble size (Babcock, 1993. Eq 4)
def risk_aversion_coef_to_risk_premium_coef(risk_aversion, gamble_size):
    return np.log(0.5 * (np.exp(-risk_aversion * gamble_size) + np.exp(risk_aversion * gamble_size))) / (risk_aversion * gamble_size)


# Calculate CARA risk aversion coefficient from risk premium coefficient and gamble size (Babcock, 1993. Eq 4)
def risk_premium_coef_to_risk_aversion_coef(risk_premium, gamble_size):
    def eqn(A):
        return np.log(0.5 * (np.exp(-A * gamble_size) + np.exp(A * gamble_size))) / (A * gamble_size) - risk_premium
    return root_scalar(eqn, bracket=[0.0000001, 100]).root


# Calculate CARA risk premium probability from CARA risk premium coefficient (Babcock, 1993. Eq 9)
def risk_premium_coef_to_risk_premium_prob(risk_premium):
    def eqn(prob):
        return  np.log((1 + 4 * np.power(prob, 2)) / (1 - 4 * np.power(prob, 2))) / np.log((1 + 2 * prob) / (1 - 2 * prob)) - risk_premium
    return root_scalar(eqn, bracket=[0.0000001, 0.4999999]).root


# Calculate CARA risk aversion coefficient from risk premium probability (Babcock, 1993. Eq 4, 9)
def risk_premium_prob_to_risk_aversion_coef(risk_premium_prob, gamble_size):
    def eqn(A):
        return (
            np.log((1 + 4 * np.power(risk_premium_prob, 2)) / (1 - 4 * np.power(risk_premium_prob, 2))) /
            np.log((1 + 2 * risk_premium_prob) / (1 - 2 * risk_premium_prob)) -
            np.log(0.5 * (np.exp(-A * gamble_size) + np.exp(A * gamble_size))) / (A * gamble_size)
        )
    return root_scalar(eqn, bracket=[0.0000001, 100]).root
