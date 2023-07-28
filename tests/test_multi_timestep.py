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

import pytest

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *
from ruv.multi_timestep import *
from ruv.data_classes import *


def test_multiple_timesteps():
    alpha = 0.1
    thresholds = np.arange(0, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)
    num_steps = 20
    ens_size = 100

    # (timesteps, ens_members)
    obs = np.random.gamma(1, 5, (num_steps, 1))
    fcsts = np.random.normal(10, 1, (num_steps, ens_size))
    refs = np.random.normal(5, 3, (num_steps, ens_size))

    data = InputData(obs, fcsts, refs)
    context = DecisionContext(None, damage_func, utility_func, thresholds, economic_model, analytical_spend)

    result = multiple_timesteps(alpha, data, context, 1)

    assert np.isclose(result.ruv, 0.0445, 1e-2)
    assert np.isclose(result.avg_fcst_ex_post, -3.399, 1e-2)
    assert np.isclose(result.avg_ref_ex_post, -3.402, 1e-2)
    assert np.isclose(result.avg_obs_ex_post, -3.340, 1e-2)
