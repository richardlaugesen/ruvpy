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

from typing import Callable, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DecisionContext:
    economic_model_params: np.ndarray
    damage_function: Callable
    utility_function: Callable
    economic_model: Callable
    analytical_spend: Callable
    decision_rule: Callable
    decision_thresholds: Optional[np.ndarray] = field(default=None)

    def validate_fields(self):
        for field_name, value in self.__dict__.items():
            if field_name != 'decision_thresholds' and value is None:
                raise ValueError(f"The field '{field_name}' cannot be None")

@dataclass
class SingleParOutput:
    ruv: float

    avg_fcst_ex_post: float
    avg_obs_ex_post: float
    avg_ref_ex_post: float

    fcst_spends: np.ndarray
    obs_spends: np.ndarray
    ref_spends: np.ndarray

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


@dataclass
class MultiParOutput:
    data: dict    # econ_par, results

    # maintains data ordered by econ_par
    def insert(self, econ_par, output):
        self.data[econ_par] = output
        self.data = {econ_par: self.data[econ_par] for econ_par in sorted(self.data)}

    # return either a 1D or 2D numpy array depending on the type of SingleParOutput field that is stored in the data dict
    def get_series(self, field):
        return np.array([getattr(v, field).tolist() if isinstance(getattr(v, field), np.ndarray) else getattr(v, field) for a, v in self.data.items()])
        
    def __init__(self):
        self.data = {}

    def to_dict(self) -> dict:
        results = {}
        results['ruv'] = self.get_series('ruv')
        results['economic_model_params'] = np.array(list(self.data.keys()))

        results['avg_fcst_ex_post'] = self.get_series('avg_fcst_ex_post')
        results['avg_ref_ex_post'] = self.get_series('avg_ref_ex_post')
        results['avg_obs_ex_post'] = self.get_series('avg_obs_ex_post')

        results['fcst_spends'] = self.get_series('fcst_spends')
        results['ref_spends'] = self.get_series('ref_spends')
        results['obs_spends'] = self.get_series('obs_spends')

        results['fcst_ex_ante'] = self.get_series('fcst_ex_ante')
        results['ref_ex_ante'] = self.get_series('ref_ex_ante')
        results['obs_ex_ante'] = self.get_series('obs_ex_ante')

        results['fcst_ex_post'] = self.get_series('fcst_ex_post')
        results['ref_ex_post'] = self.get_series('ref_ex_post')
        results['obs_ex_post'] = self.get_series('obs_ex_post')

        return results
