import sys
import numpy as np
import timeit
import statistics

sys.path.append('..')

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *
from ruv.decision_methods import *
from util import *

awrc = '405209'
name = 'Acheron River at Tagerty'
area = 629.4

start_lt=1
end_lt=30

target_unity_risk_aversion = 0.3
max_damages = 10000
damages_quantile_threshold = 0.99
damages_shape = 0.2

# should take about an hour
# parallel_nodes = 8
# num_alphas = 20
# benchmark_repeats = 15
# num_timesteps = 1000

# should take about 1 minute
parallel_nodes = 6
num_alphas = 5
benchmark_repeats = 5
num_timesteps = 100

verbose = False

alphas = np.linspace(1/num_alphas, 1-1/num_alphas, num_alphas)

target_risk_premium = risk_aversion_coef_to_risk_premium(target_unity_risk_aversion, 1)
adjusted_risk_aversion = risk_premium_to_risk_aversion_coef(target_risk_premium, max_damages)

obs, fcst, clim = load_data(awrc, start_lt, end_lt, area)
ref = clim

obs = obs[0:num_timesteps]
fcst = fcst[0:num_timesteps]
ref = ref[0:num_timesteps]

decision_definition = {
    'econ_pars': alphas,
    'target_unity_risk_aversion': target_unity_risk_aversion,
    'target_risk_premium': target_risk_premium,
    'adjusted_risk_aversion': adjusted_risk_aversion,
    'utility_function': [cara, {'A': adjusted_risk_aversion}],
    'economic_model': [cost_loss, cost_loss_analytical_spend],
    'decision_method': 'optimise_over_forecast_distribution',
    'decision_thresholds': None,
    'damage_function': [logistic, {'k': damages_shape, 'A': max_damages, 'threshold': np.nanquantile(obs, damages_quantile_threshold)}]
}

benchmark_results = timeit.repeat(
    lambda: relative_utility_value(obs, fcst, ref, decision_definition, parallel_nodes=parallel_nodes, verbose=verbose),
    number=1, repeat=benchmark_repeats)

times = np.array(benchmark_results) / num_alphas

print(f"Mean: {statistics.mean(times)} seconds")
print(f"Median: {statistics.median(times)} seconds")
print(f"Standard Deviation: {statistics.stdev(times)} seconds")

times_per_step = times/num_timesteps

print(f"Mean: {statistics.mean(times_per_step)} seconds")
print(f"Median: {statistics.median(times_per_step)} seconds")
print(f"Standard Deviation: {statistics.stdev(times_per_step)} seconds")
