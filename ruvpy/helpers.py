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
import numpy as np
from scipy.optimize import root_scalar


def probabilistic_to_deterministic_forecast(ensembles: np.ndarray, crit_thres: float) -> np.ndarray:
    """Convert an ensemble forecast to a deterministic one.

    The conversion selects the value corresponding to ``crit_thres`` of the
    exceedance distribution for each timestep.

    Args:
        ensembles: Array of ensemble forecasts with shape ``(n_times, n_members)``.
        crit_thres: Critical probability threshold used to select the
            deterministic value.

    Returns:
        Array of deterministic forecasts for each timestep.
    """
    if is_deterministic(ensembles[0]):
        raise ValueError('Forecast is already deterministic')
    return np.nanquantile(ensembles, 1 - crit_thres, axis=1)


def generate_event_freq_ref(obs: np.ndarray) -> np.ndarray:
    """Create an event-frequency reference ensemble from observations.

    This reproduces the event-frequency reference used in REV by treating
    the observation record as an ensemble for each timestep. Missing values
    are removed before tiling the observations along the time dimension to
    simplify the calculation of forecast likelihoods.

    Args:
        obs: Array of observations.

    Returns:
        2D array where each row contains the non-missing observations.
    """
    return np.tile(obs[~np.isnan(obs)], (obs.shape[0], 1))


def ecdf(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return exceedance probabilities for the supplied thresholds.

    This implementation is up to five times faster than ``statsmodels``' ECDF.
    """
    if len(thresholds) == 0:
        return thresholds
    if len(ens) == 0:
        return np.full(thresholds, np.nan)
    
    ens_sorted = np.sort(ens)
    idx = np.searchsorted(ens_sorted, thresholds)   # 3 times faster then linspace
    probs = np.arange(len(ens) + 1)/float(len(ens))
    return 1 - probs[idx]


def is_deterministic(series) -> bool:
    """Return ``True`` if the series represents a deterministic forecast."""
    if isinstance(series, np.ndarray) and len(series.shape) > 1:
        raise ValueError(
            'Forecast used for timestep should be a single value (deterministic) or a 1D array (ensemble)')

    if isinstance(series, (int, float)) or len(series) == 1:
        return True

    return False


def risk_aversion_coef_to_risk_premium(risk_aversion: float, gamble_size: float) -> float:
    """Convert a CARA risk-aversion coefficient to a risk premium.

    Calculation follows Babcock (1993, Eq. 4).
    """
    return np.log(0.5 * (np.exp(-risk_aversion * gamble_size) + np.exp(risk_aversion * gamble_size))) / (risk_aversion * gamble_size)


def risk_premium_to_risk_aversion_coef(risk_premium: float, gamble_size: float) -> float:
    """Convert a CARA risk premium to a risk-aversion coefficient.

    Calculation follows Babcock (1993, Eq. 4).
    """
    if risk_premium < 0 or risk_premium > 1:
        raise Exception('risk_premium range is 0 to 1')

    def eqn(A):
        return np.log(0.5 * (np.exp(-A * gamble_size) + np.exp(A * gamble_size))) / (A * gamble_size) - risk_premium

    return root_scalar(eqn, bracket=[1e-12, 70]).root


def risk_premium_to_prob_premium(risk_premium: float) -> float:
    """Convert a risk premium to the corresponding probability premium.

    Calculation follows Babcock (1993, Eq. 9).
    """
    if risk_premium < 0 or risk_premium > 1:
        raise Exception('risk_premium range is 0 to 1')

    if risk_premium > 0.99:
        raise Exception('scipy optimiser fails when risk_premium > 0.99')

    def eqn(prob):
        return np.log((1 + 4 * np.power(prob, 2)) / (1 - 4 * np.power(prob, 2))) / np.log((1 + 2 * prob) / (1 - 2 * prob)) - risk_premium

    return root_scalar(eqn, bracket=[1e-12, 0.49999]).root


def prob_premium_to_risk_aversion_coef(risk_premium_prob: float, gamble_size: float) -> float:
    """Convert a probability premium to a CARA risk-aversion coefficient.

    Calculation follows Babcock (1993, Eq. 4 and 9).
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
            np.log(0.5 * (np.exp(-A * gamble_size) + np.exp(A * gamble_size))) / (A * gamble_size)
        )

    return root_scalar(eqn, bracket=[1e-12, 100]).root


def nanmode(data, axis=None):
    """Return the mode while ignoring ``NaN`` values."""
    if axis is None:
        # Flatten array and remove NaN values
        non_nan_data = data[~np.isnan(data)]
        if len(non_nan_data) == 0:
            return np.nan
        unique_vals, counts = np.unique(non_nan_data, return_counts=True)
        return unique_vals[np.argmax(counts)]
    else:
        # Compute mode along the specified axis, ignoring NaN
        def mode_along_axis(subarray):
            non_nan_data = subarray[~np.isnan(subarray)]
            if len(non_nan_data) == 0:
                return np.nan
            unique_vals, counts = np.unique(non_nan_data, return_counts=True)
            return unique_vals[np.argmax(counts)]
        
        return np.apply_along_axis(mode_along_axis, axis, data)
