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

from ruv.relative_utility_value import relative_utility_value
from ruv.damage_functions import logistic_zero
from ruv.economic_models import cost_loss, cost_loss_analytical_spend
from ruv.utility_functions import cara
from ruv.decision_rules import optimise_over_forecast_distribution, critical_probability_threshold_equals_par, critical_probability_threshold_fixed


def test_relative_utility_value():

    np.random.seed(42)
    num_steps = 20
    ens_size = 100

    # (timesteps, ens_members)
    obs = np.random.gamma(1, 5, (num_steps, 1))
    obs[obs < 0] = 0

    fcsts = np.random.normal(10, 1, (num_steps, ens_size))
    fcsts[fcsts < 0] = 0

    refs = np.random.normal(5, 3, (num_steps, ens_size))
    refs[refs < 0] = 0

    decision_definition = {
        'damage_function': [logistic_zero, {'A': 1, 'k': 0.5, 'threshold': 15}],
        'utility_function': [cara, {'A': 0.3}],
        'economic_model': [cost_loss, cost_loss_analytical_spend, np.array([0.001, 0.25, 0.5, 0.75, 0.999])],
        'decision_thresholds': np.arange(0, 20, 3),
        'decision_rule': [optimise_over_forecast_distribution, None],
    }

    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'],[0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)

    decision_definition['decision_rule'] = [critical_probability_threshold_equals_par, None]
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'],[0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)

    decision_definition['decision_rule'] = [critical_probability_threshold_fixed, {'critical_probability_threshold': 0.1}]
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'], [-0.182290404, -0.0742971673, -0.467401918, -1.65026591, -616.302376], 1e-3)

    decision_definition['decision_rule'] = [critical_probability_threshold_equals_par, None]
    decision_definition['event_freq_ref'] = True
    results = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'],[-74.0584681, -0.0742971679, -0.472369878, -1.71864364, -117.108684], 1e-3)

    decision_definition = {
        'damage_function': [logistic_zero, {'A': 1, 'k': 0.5, 'threshold': 15}],
        'utility_function': [cara, {'A': 0.3}],
        'economic_model': [cost_loss, cost_loss_analytical_spend, np.array([0.001, 0.25, 0.5, 0.75, 0.999])],
        'decision_thresholds': np.arange(0, 20, 3),
        'decision_rule': [optimise_over_forecast_distribution, None],
    }

    max_val = np.max([np.nanmax(obs), np.nanmax(fcsts), np.nanmax(refs)])
    threshold_size = 5000
    decision_definition['decision_thresholds'] = np.linspace(0, max_val, threshold_size)
    many_thresholds = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)

    decision_definition['decision_thresholds'] = None
    continuous = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)
    assert np.allclose(many_thresholds['ruv'], continuous['ruv'], 0.01)
