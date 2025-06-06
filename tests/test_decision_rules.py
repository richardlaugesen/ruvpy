import numpy as np

from ruvpy.data_classes import DecisionContext
from ruvpy.economic_models import cost_loss, cost_loss_analytical_spend
from ruvpy.utility_functions import cara
from ruvpy.damage_functions import logistic_zero
from ruvpy.helpers import generate_event_freq_ref
from ruvpy.decision_rules import optimise_over_forecast_distribution, critical_probability_threshold_equals_par, critical_probability_threshold_fixed, critical_probability_threshold_max_value


def get_data(ref_equals_fcst=False, event_freq_ref=False):
    np.random.seed(42)

    fcsts = np.random.normal(10, 1, (20, 100))  # (timesteps, ens_members)
    fcsts[fcsts < 0] = 0

    obs = np.random.gamma(1, 5, (20, 1))
    obs[obs < 0] = 0

    if ref_equals_fcst:
        refs = fcsts
    else:
        refs = np.random.normal(5, 3, (20, 100))
        refs[refs < 0] = 0

        if event_freq_ref:
            refs = generate_event_freq_ref(obs)

    return obs, fcsts, refs


def get_context(decision_rule, decision_rule_params, risk_aversion=0.3):
    context_params = {
        'economic_model_params': np.array([0.25, 0.5, 0.75]),
        'utility_function': cara({'A': risk_aversion}),
        'damage_function': logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15}),
        'decision_thresholds': np.arange(0, 20, 3),
        'economic_model': cost_loss,
        'analytical_spend': cost_loss_analytical_spend,
        'decision_rule': decision_rule(decision_rule_params),
        'optimiser': {'lower_bound': 0, 'upper_bound': 2, 'tolerance': 1e-4, 'polish': True, 'seed': 42}
    }
    return DecisionContext(**context_params)


def test_optimise_over_forecast_distribution():
    context = get_context(optimise_over_forecast_distribution, None)

    # basic ensemble fcst and ref
    obs, fcsts, refs = get_data()
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [-0.18123400, -0.82382394, -2.66057767], rtol=1e-2, atol=1e-3)

    # event freq ref
    obs, fcsts, refs = get_data(event_freq_ref=True)
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [-0.180943092, -0.858109342, -2.89897125], rtol=1e-2, atol=1e-3)
    
    # ref equals fcst
    obs, fcsts, refs = get_data(ref_equals_fcst=True)
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [0.0000, 0.0000, 0.0000], rtol=1e-2, atol=1e-3)


def test_critical_probability_threshold_equals_par():
    obs, fcsts, refs = get_data()

    context = get_context(critical_probability_threshold_equals_par, None, 0)
    econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    context = get_context(optimise_over_forecast_distribution, None, 0)
    optim_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(econ_par_result.get_series('ruv'), optim_result.get_series('ruv'), rtol=1e-2, atol=1e-3)

    # context = get_context(critical_probability_threshold_equals_par, None, 0.1)
    # econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    # context = get_context(optimise_over_forecast_distribution, None, 0.1)
    # optim_result = context.decision_rule(obs, fcsts, refs, context, 1)
    # assert not np.allclose(econ_par_result.get_series('ruv'), optim_result.get_series('ruv'), rtol=1e-2, atol=1e-3)

    # context = get_context(critical_probability_threshold_equals_par, None, 5)
    # econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    # context = get_context(optimise_over_forecast_distribution, None, 5)
    # optim_result = context.decision_rule(obs, fcsts, refs, context, 1)
    # assert not np.allclose(econ_par_result.get_series('ruv'), optim_result.get_series('ruv'), rtol=1e-2, atol=1e-3)


def test_critical_probability_threshold_fixed():
    obs, fcsts, refs = get_data()
    context = get_context(critical_probability_threshold_fixed, {'critical_probability_threshold': 0.5})
    result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(result.get_series('ruv'), [-0.180944535, -0.825034187, -2.66342398], rtol=1e-2, atol=1e-3)


def test_critical_probability_threshold_max_value():
    obs, fcsts, refs = get_data()
    context = get_context(critical_probability_threshold_max_value, None)
    max_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.allclose(max_result.get_series('ruv'), [0.00008009, -0.10374633, -0.43102736], rtol=1e-2, atol=1e-3)

    context = get_context(critical_probability_threshold_equals_par, None)
    econ_par_result = context.decision_rule(obs, fcsts, refs, context, 1)
    assert np.all(max_result.get_series('ruv')[1:] >= econ_par_result.get_series('ruv')[1:])   # ignore first value because economic parameter value is extremely small

