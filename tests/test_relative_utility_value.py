# Copyright 2023 Richard Laugesen
#
# This file is part of RUV
#
# RUV is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RUV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RUV.  If not, see <https://www.gnu.org/licenses/>.

from statsmodels.distributions.empirical_distribution import ECDF
import pytest

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *

# TODO: Now tests are written, confirm that the numbers in here are actually what we excpect, could calculate manually

def test_ecdf_numpy():
    ens = np.array([])
    thresholds = np.array([])
    assert np.array_equal(ecdf_numpy(ens, thresholds), np.array([]))

    ens = np.array([1, 2, 3, 4, 5])
    thresholds = np.array([])
    assert np.array_equal(ecdf_numpy(ens, thresholds), np.array([]))

    ens = np.array([])
    thresholds = np.array([1, 3, 5])
    assert np.all(np.isnan(ecdf_numpy(ens, thresholds)))

    ens = np.array([1, 2, 3, 4, 5])
    thresholds = np.array([0, 3])
    assert np.array_equal(ecdf_numpy(ens, thresholds), np.array([1, 0.6]))

    ens = np.random.normal(10, 1, 1000)
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(ecdf_numpy(ens, thresholds),
                      np.subtract(1, ECDF(ens, 'left')(thresholds)), 1e-3)


def test_calc_likelihoods():
    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    
    thresholds = np.arange(5, 15, 1)
    assert np.allclose(calc_likelihoods(ens, thresholds),
                       np.array([0, 0, 0.01, 0.16, 0.37, 0.35, 0.11, 0, 0, 0]), 1e-1)

    with pytest.raises(ValueError):
        calc_likelihoods(3, thresholds)

    with pytest.raises(ValueError):
        calc_likelihoods(3.2, thresholds)

    with pytest.raises(ValueError):
        calc_likelihoods([3], thresholds)

    assert np.array_equal(calc_likelihoods([7], thresholds), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(calc_likelihoods(5, thresholds), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(calc_likelihoods(9, thresholds), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert np.array_equal(calc_likelihoods(30, thresholds), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    with pytest.raises(ValueError):
        ens[20:60] = np.nan
        #ens = ens[~np.isnan(ens)]
        calc_likelihoods(ens, thresholds)

    with pytest.raises(ValueError):
        calc_likelihoods(None, thresholds)

    with pytest.raises(ValueError):
        calc_likelihoods(None, None)

    thresholds = None
    assert np.array_equal(calc_likelihoods(ens, thresholds),
                          np.full(100, 1e-2))
    

def test_realised_threshold():
    thresholds = np.array([0, 3, 6])
    assert np.equal(realised_threshold(0.5, thresholds), 0)
    assert np.equal(realised_threshold(3, thresholds), 3)
    assert np.equal(realised_threshold(3.5, thresholds), 3)
    assert np.equal(realised_threshold(6, thresholds), 6)
    assert np.equal(realised_threshold(7, thresholds), 6)

    with pytest.raises(ValueError):
        realised_threshold(0.1, [1, 2, 3])

    with pytest.raises(ValueError):
        values = [0.5, 3, 3.5, 6, 7]
        realised_threshold(values, thresholds)

    assert np.equal(realised_threshold(42, None), 42)


def test_probabilistic_to_deterministic_forecast():
    np.random.seed(42)
    fcst_ens = np.random.normal(10, 1, (5, 1000))  # (timesteps, ens_members)

    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([10.02530061, 10.06307713,  9.99974924, 10.00018457,  9.98175801]), 1e-5)
    
    assert np.alltrue(probabilistic_to_deterministic_forecast(fcst_ens, 0.1) > probabilistic_to_deterministic_forecast(fcst_ens, 0.9))
    
    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, -1)

    with pytest.raises(ValueError):
        probabilistic_to_deterministic_forecast(fcst_ens, 2)

    fcst_ens = np.random.normal(10, 1, (1, 1000))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        np.array([9.95717293]), 1e-5)
    
    fcst_ens = np.random.normal(10, 1, (10, 1))
    assert np.allclose(
        probabilistic_to_deterministic_forecast(fcst_ens, 0.5),
        fcst_ens.T, 1e-5)


def test_generate_event_freq_ref():
    obs = np.random.gamma(1, 2, 1000)
    idx = np.random.randint(0, obs.size, 100)
    obs[idx] = np.nan
    ref = generate_event_freq_ref(obs)
    assert np.isclose(np.mean(ref), np.nanmean(obs), 1e-5)
    assert np.array_equal(ref.shape, (obs.shape[0], obs.shape[0] - np.sum(np.isnan(obs))))


def test_ex_ante_utility():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})
    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = calc_likelihoods(ens, thresholds)

    alpha = 0.1; spend = 0.052
    assert np.isclose(
        ex_ante_utility(spend, probs, thresholds, alpha, economic_model, damage_func, utility_func),
        -3.386, 1e-2)
    
    alpha = 0.7; spend = 0.052
    assert np.isclose(
        ex_ante_utility(spend, probs, thresholds, alpha, economic_model, damage_func, utility_func),
        -3.386, 1e-2)
    
    alpha = 0.1; spend = 3
    assert np.isclose(
        ex_ante_utility(spend, probs, thresholds, alpha, economic_model, damage_func, utility_func),
        -8.199, 1e-2)    


def test_ex_post_utility():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})
    
    occured = thresholds[10]; alpha = 0.1; spend = 0.052
    assert np.isclose(
        ex_post_utility(occured, spend, alpha, economic_model, damage_func, utility_func),
        -3.386, 1e-2)
    
    occured = thresholds[10]; alpha = 0.7; spend = 0.052
    assert np.isclose(
        ex_post_utility(occured, spend, alpha, economic_model, damage_func, utility_func),
        -3.847, 1e-2)
    
    occured = thresholds[10]; alpha = 0.1; spend = 3
    assert np.isclose(
        ex_post_utility(occured, spend, alpha, economic_model, damage_func, utility_func),
        -8.199, 1e-2)    


def test_find_spend():
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)
    ens = np.random.normal(10, 1, 100)
    probs = calc_likelihoods(ens, thresholds)
    alpha = 0.1
    assert np.isclose(
        find_spend(ens, probs, thresholds, alpha, economic_model, analytical_spend, damage_func, utility_func),
        0.012, 1e-1)

    det = 7.2
    probs = None
    alpha = 0.1
    assert np.isclose(
        find_spend(det, probs, thresholds, alpha, economic_model, analytical_spend, damage_func, utility_func),
        0.0018, 1e-3)


def test_single_timestep():
    t = 1
    ob = 10
    alpha = 0.1    
    thresholds = np.arange(5, 20, 1)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)    
    fcst = np.random.normal(10, 1, 100)
    fcst_probs = calc_likelihoods(fcst, thresholds)

    ref = np.random.normal(5, 3, 100)
    ref_probs = calc_likelihoods(ref, thresholds)

    results = single_timestep(t, ob, thresholds, alpha, economic_model, analytical_spend, damage_func, utility_func, fcst, fcst_probs, ref, ref_probs)
    
    assert np.isclose(results['obs_spends'], 0.0076, 1e-2) 
    assert np.isclose(results['fcst_spends'], 0.012, 1e-2) 
    assert np.isclose(results['ref_spends'], 0.0076, 1e-2) 


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
    fcst_likelihoods = all_likelihoods(obs, fcsts, thresholds)
    refs = np.random.normal(5, 3, (num_steps, ens_size))
    ref_likelihoods = all_likelihoods(obs, refs, thresholds)

    result = multiple_timesteps(alpha, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, 1)

    assert np.isclose(result['ruv'], 0.0445, 1e-2)
    assert np.isclose(result['fcst_avg_ex_post'], -3.399, 1e-2)
    assert np.isclose(result['obs_avg_ex_post'], -3.340, 1e-2)
    assert np.isclose(result['ref_avg_ex_post'], -3.402, 1e-2)


def test_multiple_alpha():

    alphas = np.array([0.001, 0.25, 0.5, 0.75, 0.999])
    thresholds = np.arange(0, 20, 3)
    economic_model = cost_loss
    analytical_spend = cost_loss_analytical_spend
    damage_func = logistic_zero({'A': 1, 'k': 0.5, 'threshold': 15})
    utility_func = cara({'A': 0.3})

    np.random.seed(42)
    num_steps = 20
    ens_size = 100

    # (timesteps, ens_members)
    obs = np.random.gamma(1, 5, (num_steps, 1))
    obs[obs < 0] = 0

    fcsts = np.random.normal(10, 1, (num_steps, ens_size))
    fcsts[fcsts < 0] = 0
    
    fcst_likelihoods = all_likelihoods(obs, fcsts, thresholds)
    refs = np.random.normal(5, 3, (num_steps, ens_size))
    refs[refs < 0] = 0
    ref_likelihoods = all_likelihoods(obs, refs, thresholds)    

    # TODO: update this test so results discriminate between the decision_method cases

    decision_method = 'optimise_over_forecast_distribution'
    result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    assert np.allclose(
        result['ruv'],
        [0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)

    decision_method = 'critical_probability_threshold_equals_alpha'
    result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    assert np.allclose(
        result['ruv'],
        [0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)

    refs = generate_event_freq_ref(obs)
    ref_likelihoods = all_likelihoods(obs, refs, thresholds)    

    decision_method = 'optimise_over_forecast_distribution'
    result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    assert np.allclose(
        result['ruv'],
        [-74.0584681, -0.0742971679, -0.472369878, -1.71864364, -117.108684], 1e-3)

    decision_method = 'critical_probability_threshold_equals_alpha'
    result = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    assert np.allclose(
        result['ruv'],
        [-74.0584681, -0.0742971679, -0.472369878, -1.71864364, -117.108684], 1e-3)

    # two methods are equivilent when risk aversion is small
    utility_func = cara({'A': 0.1})
    decision_method = 'optimise_over_forecast_distribution'
    result_1 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    result_1 = result_1
    decision_method = 'critical_probability_threshold_equals_alpha'
    result_2 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    assert np.allclose(result_1['ruv'], result_2['ruv'], 1e-3)    

    # two methods not equivilent when risk aversion is high
    utility_func = cara({'A': 5})
    decision_method = 'optimise_over_forecast_distribution'
    result_1 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    decision_method = 'critical_probability_threshold_equals_alpha'
    result_2 = multiple_alpha(alphas, obs, fcsts, refs, fcst_likelihoods, ref_likelihoods, thresholds, economic_model, analytical_spend, damage_func, utility_func, decision_method, 1)
    assert not np.allclose(result_1['ruv'], result_2['ruv'], 1e-3)


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
        'alphas': np.array([0.001, 0.25, 0.5, 0.75, 0.999]),
        'damage_function': [logistic_zero, {'A': 1, 'k': 0.5, 'threshold': 15}],
        'utility_function': [cara, {'A': 0.3}],
        'economic_model': [cost_loss, cost_loss_analytical_spend],
        'decision_thresholds': np.arange(0, 20, 3),
        'decision_method': 'optimise_over_forecast_distribution'
    }

    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(
        results['ruv'],
        [0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)
    
    decision_definition['decision_method'] = 'critical_probability_threshold_equals_alpha'
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(
        results['ruv'],
        [0.184053111, -0.0742971672, -0.467401918, -1.65026591, -117.108686], 1e-3)
       
    decision_definition['decision_method'] = 'critical_probability_threshold_fixed'
    decision_definition['critical_probability_threshold'] = 0.1
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(
        results['ruv'],
        [0.04111241, -0.06836699, -0.22266255, -0.46356886, -0.9073991], 1e-3)

    decision_definition['decision_method'] = 'critical_probability_threshold_equals_alpha'
    refs = None
    results = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.allclose(
        results['ruv'],
        [-74.0584681, -0.0742971679, -0.472369878, -1.71864364, -117.108684], 1e-3)
    
    decision_definition = {
        'alphas': np.array([0.001, 0.25, 0.5, 0.75, 0.999]),
        'damage_function': [logistic_zero, {'A': 1, 'k': 0.5, 'threshold': 15}],
        'utility_function': [cara, {'A': 0.3}],
        'economic_model': [cost_loss, cost_loss_analytical_spend],
        'decision_thresholds': np.arange(0, 20, 3)
    }
    results_default_method = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)

    decision_definition['decision_method'] = 'optimise_over_forecast_distribution'
    results_defined_method = relative_utility_value(obs, fcsts, refs, decision_definition, parallel_nodes=2)
    assert np.array_equal(results_default_method['ruv'], results_defined_method['ruv'])


def test_risk_aversion_coef_to_risk_premium():
    risk_aversions = [0.3, 1, 5]
    gamble_size = 1
    expected_result = [0.1478, 0.4338, 0.8614]
    for i, risk_aversion in enumerate(risk_aversions):
        assert np.isclose(risk_aversion_coef_to_risk_premium(risk_aversion, gamble_size), expected_result[i], 1e-3)

    risk_aversions = [0.3, 1, 5]
    gamble_size = 10
    expected_result = [0.7698, 0.9307, 0.98614]
    for i, risk_aversion in enumerate(risk_aversions):
        assert np.isclose(risk_aversion_coef_to_risk_premium(risk_aversion, gamble_size), expected_result[i], 1e-3)


def test_risk_premium_to_risk_aversion_coef():
    risk_premiums = [0.1478, 0.4338, 0.8614]
    gamble_size = 1
    expected_result = [0.3, 1, 5]
    for i, risk_premium in enumerate(risk_premiums):
        assert np.isclose(risk_premium_to_risk_aversion_coef(risk_premium, gamble_size), expected_result[i], 1e-3) 

    risk_premiums = [0.7698, 0.9307, 0.98614]
    gamble_size = 10
    expected_result = [0.3, 1, 5]
    for i, risk_premium in enumerate(risk_premiums):
        assert np.isclose(risk_premium_to_risk_aversion_coef(risk_premium, gamble_size), expected_result[i], 1e-3) 


def test_risk_premium_to_prob_premium():
    risk_premiums = [0.1478, 0.4338, 0.8614]
    expected_result = [0.0744, 0.2311, 0.4933]
    for i, risk_premium in enumerate(risk_premiums):
        assert np.isclose(risk_premium_to_prob_premium(risk_premium), expected_result[i], 1e-3)      

    risk_premium_probs = [0.283559, 0.662501, 0.930685]
    expected_result = [0.145656, 0.380797, 0.499955]
    for i, risk_premium_prob in enumerate(risk_premium_probs):
        assert np.isclose(risk_premium_to_prob_premium(risk_premium_prob), expected_result[i], 1e-3)  

    risk_premium_prob = 0.7698
    expected_result = 0.452574
    assert np.isclose(risk_premium_to_prob_premium(risk_premium_prob), expected_result, 1e-3)  

    risk_premium_prob = 0.9999
    with pytest.raises(Exception):
        risk_premium_to_prob_premium(risk_premium_prob)


def test_prob_premium_to_risk_aversion_coef():
    risk_premium_probs = [0.0744, 0.2311, 0.4933]
    gamble_size = 1
    expected_result = [0.3, 1, 5]
    for i, risk_premium_prob in enumerate(risk_premium_probs):
        assert np.isclose(prob_premium_to_risk_aversion_coef(risk_premium_prob, gamble_size), expected_result[i], 1e-3)
     
    risk_premium_probs = [0.145656, 0.380797, 0.499955]
    gamble_size = 2
    expected_result = [0.3, 1, 5]
    for i, risk_premium_prob in enumerate(risk_premium_probs):
        assert np.isclose(prob_premium_to_risk_aversion_coef(risk_premium_prob, gamble_size), expected_result[i], 1e-3)   

    risk_premium_prob = 0.93
    gamble_size = 10
    with pytest.raises(Exception):
        prob_premium_to_risk_aversion_coef(risk_premium_prob, gamble_size)
