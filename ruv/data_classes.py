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

from typing import Callable, Optional
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InputData:
    obs: np.ndarray
    fcsts: np.ndarray
    refs: np.ndarray


@dataclass(frozen=True)
class DecisionContext:
    alphas: np.ndarray
    damage_function: Callable
    utility_function: Callable
    decision_thresholds: np.ndarray
    economic_model: Callable
    analytical_spend: Callable
    crit_prob_thres: Optional[float] = None
    event_freq_ref: bool = False


@dataclass
class SingleAlphaOutput:
    ruv: float
    avg_fcst_ex_post: float
    avg_obs_ex_post: float
    avg_ref_ex_post: float
    fcst_spends: np.ndarray
    obs_spends: np.ndarray
    ref_spends: np.ndarray
    fcst_likelihoods: np.ndarray
    ref_likelihoods: np.ndarray
    fcst_ex_post: np.ndarray
    obs_ex_post: np.ndarray
    ref_ex_post: np.ndarray
    fcst_ex_ante: np.ndarray
    obs_ex_ante: np.ndarray
    ref_ex_ante: np.ndarray

    def __init__(self, obs_size: int):
        self.ruv = np.nan
        self.avg_fcst_ex_post = np.nan
        self.avg_obs_ex_post = np.nan
        self.avg_ref_ex_post = np.nan
        self.fcst_spends, self.obs_spends, self.ref_spends = np.full((3, obs_size), np.nan)
        self.fcst_ex_ante, self.obs_ex_ante, self.ref_ex_ante = np.full((3, obs_size), np.nan)
        self.fcst_ex_post, self.obs_ex_post, self.ref_ex_post = np.full((3, obs_size), np.nan)
        self.fcst_likelihoods, self.ref_likelihoods = np.full((2, obs_size), np.nan)


@dataclass
class MultiAlphaOutput:
    data: dict[float, SingleAlphaOutput]    # alpha, results

    # maintains data ordered by alpha
    def insert(self, alpha, output):
        self.data[alpha] = output
        self.data = {alpha: self.data[alpha] for alpha in sorted(self.data)}

    # return either a 1D or 2D numpy array depending on the type of SingleAlphaOutput field that is stored in the data dict
    def get_series(self, field):
        return np.array([getattr(v, field).tolist() if isinstance(getattr(v, field), np.ndarray) else getattr(v, field) for a, v in self.data.items()])
        
    def __init__(self):
        self.data = {}
