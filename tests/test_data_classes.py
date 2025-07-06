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
import numpy as np
from ruvpy.data_classes import DecisionContext
from ruvpy.damage_functions import logistic_zero
from ruvpy.utility_functions import cara
from ruvpy.economic_models import cost_loss, cost_loss_analytical_spend
from ruvpy.decision_rules import optimise_over_forecast_distribution

def test_decision_context_default_optimiser():
    context = DecisionContext(
        economic_model_params=np.array([0.1]),
        damage_function=logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15}),
        utility_function=cara({'A': 0.3}),
        economic_model=cost_loss,
        analytical_spend=cost_loss_analytical_spend,
        decision_rule=optimise_over_forecast_distribution,
        decision_thresholds=None
    )
    assert context.optimiser == {
        'lower_bound': None,
        'upper_bound': None,
        'tolerance': 1E-4,
        'polish': True,
        'seed': None
    }
