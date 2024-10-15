# Copyright 2024 Richard Laugesen

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

from ruv.multi_timestep import multiple_timesteps
from ruv.data_classes import DecisionContext
from ruv.damage_functions import logistic_zero
from ruv.economic_models import cost_loss, cost_loss_analytical_spend
from ruv.decision_rules import optimise_over_forecast_distribution
from ruv.utility_functions import cara

def get_context():
    context_fields = {
        'economic_model_params': np.array([0.1]),
        'damage_function': logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15}),
        'utility_function': cara({'A': 0.3}),
        'decision_thresholds': np.arange(0, 20, 1),
        'economic_model': cost_loss,
        'analytical_spend': cost_loss_analytical_spend,
        'decision_rule': optimise_over_forecast_distribution
    }
    return DecisionContext(**context_fields)


def test_multiple_timesteps():
    context = get_context()

    np.random.seed(42)
    num_steps = 20
    ens_size = 100

    # (timesteps, ens_members)
    obs = np.random.gamma(1, 5, (num_steps, 1))
    fcsts = np.random.normal(10, 1, (num_steps, ens_size))
    refs = np.random.normal(5, 3, (num_steps, ens_size))

    econ_par = context.economic_model_params[0]

    result = multiple_timesteps(obs, fcsts, refs, econ_par, context, 1)

    assert np.isclose(result.ruv, 0.0445, 1e-2)
    assert np.isclose(result.avg_fcst_ex_post, -3.399, 1e-2)
    assert np.isclose(result.avg_ref_ex_post, -3.402, 1e-2)
    assert np.isclose(result.avg_obs_ex_post, -3.340, 1e-2)
