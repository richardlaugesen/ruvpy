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
from scipy.optimize import root_scalar


def generate_event_freq_ref(obs: np.ndarray) -> np.ndarray:
    """
    Generate an 'event frequency' reference distribution for each timestep.
    The reference distribution is simply the observed record with any missing values dropped.
    This is then tiled to match the original observation shape.

    Parameters
    ----------
    obs : np.ndarray
        The observed data as a 1D numpy array.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row is the observed data with missing values dropped.
        The number of rows is equal to the number of elements in the input array.

    Notes
    -----
    NaN values in the input array are considered as missing values and are dropped.
    """
    return np.tile(obs[~np.isnan(obs)], (obs.shape[0], 1))


def ecdf(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Compute the empirical cumulative distribution function (ECDF) for a given ensemble 
    and set of thresholds. The ECDF represents the probability that a random variable 
    is less than or equal to a certain value.

    Parameters
    ----------
    ens : ndarray
        The ensemble for which to compute the ECDF. This should be a 1D numpy array 
        of ensemble members.

    thresholds : ndarray
        The thresholds at which to compute the ECDF. This should be a 1D numpy array 
        of values.

    Returns
    -------
    ndarray
        A numpy array of the same shape as `thresholds`, where each element is the 
        probability that a randomly chosen member of `ens` is less than or equal to 
        the corresponding value in `thresholds`.

    Notes
    -----
    This implementation is around 5 times faster then statsmodels ECDF
    """
    ens_sorted = np.sort(ens)
    idx = np.searchsorted(ens_sorted, thresholds)
    # 3 times fast then linspace
    probs = np.arange(ens.size + 1)/float(ens.size)
    return 1 - probs[idx]


def is_deterministic_timestep(series) -> bool:
    """
    Check if the given series represents a deterministic timestep.

    Args:
        series: The series to check.

    Returns:
        bool: True if the series represents a deterministic timestep, False otherwise.

    Raises:
        ValueError: If the series is a 2D array.
    """
    if isinstance(series, np.ndarray) and len(series.shape) > 1:
        raise ValueError(
            'Forecast used for timestep should be a single value (deterministic) or a 1D array (ensemble)')

    if isinstance(series, (int, float)) or len(series) == 1:
        return True

    return False


def risk_aversion_coef_to_risk_premium(risk_aversion: float, gamble_size: float) -> float:
    """
    Calculate CARA risk premium from risk aversion coefficient and gamble size (Babcock, 1993. Eq 4)

    Args:
        risk_aversion (float): Risk aversion coefficient
        gamble_size (float): Gamble size

    Returns:
        float: Risk premium
    """
    return np.log(0.5 * (np.exp(-risk_aversion * gamble_size) + np.exp(risk_aversion * gamble_size))) / (risk_aversion * gamble_size)


def risk_premium_to_risk_aversion_coef(risk_premium: float, gamble_size: float) -> float:
    """
    Calculate CARA risk aversion coefficient from risk premium and gamble size (Babcock, 1993. Eq 4)

    Args:
        risk_premium (float): Risk premium
        gamble_size (float): Gamble size

    Raises:
        Exception: If risk_premium is not in the range of 0 to 1

    Returns:
        float: Risk aversion coefficient
    """
    if risk_premium < 0 or risk_premium > 1:
        raise Exception('risk_premium range is 0 to 1')

    def eqn(A):
        return np.log(0.5 * (np.exp(-A * gamble_size) + np.exp(A * gamble_size))) / (A * gamble_size) - risk_premium

    return root_scalar(eqn, bracket=[0.0000001, 100]).root


def risk_premium_to_prob_premium(risk_premium: float) -> float:
    """
    Calculate CARA probability premium from risk premium (Babcock, 1993. Eq 9)

    Args:
        risk_premium (float): Risk premium

    Raises:
        Exception: If risk_premium is not in the range of 0 to 1
        Exception: If risk_premium is greater than 0.99

    Returns:
        float: Probability premium
    """
    if risk_premium < 0 or risk_premium > 1:
        raise Exception('risk_premium range is 0 to 1')

    if risk_premium > 0.99:
        raise Exception('scipy optimiser fails when risk_premium > 0.99')

    def eqn(prob):
        return np.log((1 + 4 * np.power(prob, 2)) / (1 - 4 * np.power(prob, 2))) / np.log((1 + 2 * prob) / (1 - 2 * prob)) - risk_premium

    return root_scalar(eqn, bracket=[0.0000001, 0.49999]).root


def prob_premium_to_risk_aversion_coef(risk_premium_prob: float, gamble_size: float) -> float:
    """
    Calculate CARA risk aversion coefficient from probability premium (Babcock, 1993. Eq 4, 9)

    Args:
        risk_premium_prob (float): Probability premium
        gamble_size (float): Gamble size

    Raises:
        Exception: If risk_premium_prob is not in the range of 0 to 0.5
        Exception: If risk_premium_prob is greater than 0.49999

    Returns:
        float: Risk aversion coefficient
    """
    if risk_premium_prob < 0 or risk_premium_prob > 0.5:
        raise Exception('risk_premium_prob range is 0 to 0.5')

    if risk_premium_prob > 0.49999:
        raise Exception(
            'scipy optimiser fails when risk_premium_prob > 0.49999')

    def eqn(A):
        return (
            np.log((1 + 4 * np.power(risk_premium_prob, 2)) / (1 - 4 * np.power(risk_premium_prob, 2))) /
            np.log((1 + 2 * risk_premium_prob) / (1 - 2 * risk_premium_prob)) -
            np.log(0.5 * (np.exp(-A * gamble_size) +
                   np.exp(A * gamble_size))) / (A * gamble_size)
        )

    return root_scalar(eqn, bracket=[0.0000001, 100]).root
