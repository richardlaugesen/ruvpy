import numpy as np

from ruvpy.relative_utility_value import relative_utility_value
from ruvpy.damage_functions import logistic_zero
from ruvpy.economic_models import cost_loss, cost_loss_analytical_spend
from ruvpy.utility_functions import cara
from ruvpy.decision_rules import optimise_over_forecast_distribution, critical_probability_threshold_equals_par, critical_probability_threshold_fixed


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
        'economic_model': [cost_loss, cost_loss_analytical_spend, np.array([0.25, 0.5, 0.75])],
        'decision_thresholds': np.arange(0, 20, 3),
        'decision_rule': [optimise_over_forecast_distribution, None],
        'optimiser': {'lower_bound': 0, 'upper_bound': 2, 'tolerance': 1e-4, 'polish': True, 'seed': 42}
    }

    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'], [-0.07396961, -0.46616302, -1.64862845], rtol=1e-2, atol=1e-3)
    
    decision_definition['decision_rule'] = [critical_probability_threshold_equals_par, None]
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'],[-0.07412637, -0.46687758, -1.65032088], rtol=1e-2, atol=1e-3)

    decision_definition['decision_rule'] = [critical_probability_threshold_fixed, {'critical_probability_threshold': 0.1}]
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'], [-0.0742971519, -0.467401618, -1.65025176], rtol=1e-2, atol=1e-3)

    decision_definition['decision_rule'] = [critical_probability_threshold_equals_par, None]
    decision_definition['event_freq_ref'] = True
    results = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)
    assert np.allclose(results['ruv'],[-0.0742971679, -0.472369878, -1.7186436], rtol=1e-2, atol=1e-3)

    decision_definition = {
        'damage_function': [logistic_zero, {'A': 1, 'k': 0.5, 'threshold': 15}],
        'utility_function': [cara, {'A': 0.3}],
        'economic_model': [cost_loss, cost_loss_analytical_spend, np.array([0.25, 0.5, 0.75])],
        'decision_thresholds': np.arange(0, 20, 3),
        'decision_rule': [optimise_over_forecast_distribution, None],
        'optimiser': {'lower_bound': 0, 'upper_bound': 2, 'tolerance': 1e-4, 'polish': True, 'seed': 42}
    }

    max_val = np.max([np.nanmax(obs), np.nanmax(fcsts), np.nanmax(refs)])
    threshold_size = 5000
    decision_definition['decision_thresholds'] = np.linspace(0, max_val, threshold_size)
    many_thresholds = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)

    decision_definition['decision_thresholds'] = None
    continuous = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)
    assert np.allclose(many_thresholds['ruv'], continuous['ruv'], rtol=1e-2, atol=1e-3)

    decision_definition.pop('optimiser')
    continuous_no_opt = relative_utility_value(obs, fcsts, None, decision_definition, parallel_nodes=2)
    assert np.allclose(continuous['ruv'], continuous_no_opt['ruv'], rtol=1e-2, atol=1e-2)
