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
from scipy import interpolate

# Functional currying is used to allow a pre-paramterised damage functions to be passed into RUV


# Logistic-based damages, S-shaped damages with max A
def logistic(params):
    A, k, threshold = params['A'], params['k'], params['threshold']

    def damages(magnitude):
        return np.divide(A, np.add(1, np.exp(np.multiply(-k, np.subtract(magnitude, threshold)))))
    
    return damages


# Logistic-based damages but with damages=0 when magnitude=0
def logistic_zero(params):
    logistic_curry = logistic(params)

    def damages(magnitude):
        damages = logistic_curry(magnitude)
        try:
            damages[magnitude == 0] = 0
        except TypeError:
            damages = 0 if magnitude == 0 else damages
        return damages

    return damages


# Binary damages, 1 above a threshold, 0 below
def binary(params):
    threshold, max_loss, min_loss = params['threshold'], params['max_loss'], params['min_loss']

    def damages(magnitude):
        ge = np.greater_equal(magnitude, threshold)
        damages = np.empty(magnitude.shape)
        damages[ge] = max_loss
        damages[~ge] = min_loss
        return damages
    
    return damages


# Linear damages
def linear(params):
    slope, intercept = params['slope'], params['intercept']

    def damages(magnitude):
        return slope * magnitude + intercept
    
    return damages


# User defined damages by a linear interpolation over a set of points (list of tuples)
def user_defined(params):

    if 'interpolator' in params:
        inter = params['interpolator']
    else:
        points = params['points']
        inter = user_defined_interpolator(points)     

    def damages(magnitude):
        return inter(magnitude)
    
    return damages


# Convenience function for investigating interpolator used by user_defined
def user_defined_interpolator(points):
    user_flows, user_damages = zip(*points)
    extrapolate_value = user_damages[-1]
    inter = interpolate.interp1d(user_flows, user_damages, kind='linear', fill_value=extrapolate_value, bounds_error=False)
    return inter
