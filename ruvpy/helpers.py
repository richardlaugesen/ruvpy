# Copyright 2024 RUVPY Developers

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


def probabilistic_to_deterministic_forecast(ensembles: np.ndarray, crit_thres: float) -> np.ndarray:
    if is_deterministic(ensembles[0]):
        raise ValueError('Forecast is already deterministic')
    return np.nanquantile(ensembles, 1 - crit_thres, axis=1)


# Can reproduce the behaviour of event frequency reference used in REV
# using the RUV expected utility approach with optimisation over whole forecast
# distribution method if an ensemble for each timestep is used
# which is simply the observation record. NA are dropped to simplify
# calculation of forecast likelihoods
def generate_event_freq_ref(obs: np.ndarray) -> np.ndarray:
    return np.tile(obs[~np.isnan(obs)], (obs.shape[0], 1))


# Around 5 times faster than statsmodels ECDF
def ecdf(ens: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    ens_sorted = np.sort(ens)
    idx = np.searchsorted(ens_sorted, thresholds)   # 3 times faster then linspace
    probs = np.arange(len(ens) + 1)/float(len(ens))
    return 1 - probs[idx]


def is_deterministic(series) -> bool:
    if isinstance(series, np.ndarray) and len(series.shape) > 1:
        raise ValueError(
            'Forecast used for timestep should be a single value (deterministic) or a 1D array (ensemble)')

    if isinstance(series, (int, float)) or len(series) == 1:
        return True

    return False


# Calculate CARA risk premium from risk aversion coefficient and gamble size (Babcock, 1993. Eq 4)
def risk_aversion_coef_to_risk_premium(risk_aversion: float, gamble_size: float) -> float:
    return np.log(0.5 * (np.exp(-risk_aversion * gamble_size) + np.exp(risk_aversion * gamble_size))) / (risk_aversion * gamble_size)


# Calculate CARA risk aversion coefficient from risk premium and gamble size (Babcock, 1993. Eq 4)
def risk_premium_to_risk_aversion_coef(risk_premium: float, gamble_size: float) -> float:
    if risk_premium < 0 or risk_premium > 1:
        raise Exception('risk_premium range is 0 to 1')

    def eqn(A):
        return np.log(0.5 * (np.exp(-A * gamble_size) + np.exp(A * gamble_size))) / (A * gamble_size) - risk_premium

    return root_scalar(eqn, bracket=[0.0000001, 100]).root


# Calculate CARA probability premium from risk premium (Babcock, 1993. Eq 9)
def risk_premium_to_prob_premium(risk_premium: float) -> float:
    if risk_premium < 0 or risk_premium > 1:
        raise Exception('risk_premium range is 0 to 1')

    if risk_premium > 0.99:
        raise Exception('scipy optimiser fails when risk_premium > 0.99')

    def eqn(prob):
        return np.log((1 + 4 * np.power(prob, 2)) / (1 - 4 * np.power(prob, 2))) / np.log((1 + 2 * prob) / (1 - 2 * prob)) - risk_premium

    return root_scalar(eqn, bracket=[0.0000001, 0.49999]).root


# Calculate CARA risk aversion coefficient from probability premium (Babcock, 1993. Eq 4, 9)
def prob_premium_to_risk_aversion_coef(risk_premium_prob: float, gamble_size: float) -> float:
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

    return root_scalar(eqn, bracket=[0.0000001, 100]).root

