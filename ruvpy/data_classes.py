# Copyright 2023–2025 Richard Laugesen
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
from typing import Callable, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DecisionContext:
    """Container for all functions and parameters required for an RUV run."""
    economic_model_params: np.ndarray
    damage_function: Callable
    utility_function: Callable
    economic_model: Callable
    analytical_spend: Callable
    decision_rule: Callable
    decision_thresholds: np.ndarray = field(default=None)
    optimiser: dict = field(default_factory=lambda: {
        'lower_bound': None,
        'upper_bound': None,
        'tolerance': 1E-4,
        'polish': True,
        'seed': None
    })
    
    def validate_fields(self):
        """Ensure all mandatory fields are set."""
        for field_name, value in self.__dict__.items():
            if field_name != 'decision_thresholds' and value is None:
                raise ValueError(f"The field '{field_name}' cannot be None")

@dataclass
class SingleParOutput:
    """Output data for a single economic parameter value."""
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
        """Create empty arrays for storing results."""
        self.ruv = np.nan
        self.avg_fcst_ex_post = np.nan
        self.avg_obs_ex_post = np.nan
        self.avg_ref_ex_post = np.nan
        self.fcst_spends, self.obs_spends, self.ref_spends = np.full((3, obs_size), np.nan)
        self.fcst_ex_ante, self.obs_ex_ante, self.ref_ex_ante = np.full((3, obs_size), np.nan)
        self.fcst_ex_post, self.obs_ex_post, self.ref_ex_post = np.full((3, obs_size), np.nan)


@dataclass
class MultiParOutput:
    """Container for outputs over multiple economic parameter values."""

    data: dict    # econ_par, results
    def insert(self, econ_par, output):
        """Insert results keyed by economic parameter and maintain ordering."""
        self.data[econ_par] = output
        self.data = {econ_par: self.data[econ_par] for econ_par in sorted(self.data)}

    def get_series(self, field):
        """Return results for ``field`` as a 1D or 2D ``numpy`` array."""
        return np.array([
            getattr(v, field).tolist() if isinstance(getattr(v, field), np.ndarray) else getattr(v, field)
            for a, v in self.data.items()
        ])
        
    def __init__(self):
        """Initialise an empty results container."""
        self.data = {}

    def to_dict(self) -> dict:
        """Convert stored results to a dictionary of ``numpy`` arrays."""
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
