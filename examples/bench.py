import sys
import numpy as np
import timeit
import statistics
import pandas as pd

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

num_alphas = 10
benchmark_repeats = 20
num_timesteps = 200

# should take about 1 minute
#num_alphas = 5
#benchmark_repeats = 5
#num_timesteps = 100
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

results = {}

print('Starting benchmark')

for cores in range(1, 9):
    benchmark_results = timeit.repeat(
        lambda: relative_utility_value(obs, fcst, ref, decision_definition, parallel_nodes=cores, verbose=verbose),
        number=1, repeat=benchmark_repeats)

    times = np.array(benchmark_results) / num_alphas

    print('\n%d timesteps with cores=%d' % (num_timesteps, cores))
    print('Mean time per alpha: %.4f seconds +/- %.4f' % (statistics.mean(times), statistics.stdev(times)))
    print('Mean time per alpha per timestep: %.4f seconds' % (statistics.mean(times) / num_timesteps))
    
    if cores > 1:
        print('Speedup factor: %.4f (vs %d)' % (statistics.mean(results[1]) / statistics.mean(times), cores))

    results[cores] = times

df = pd.DataFrame(results)
df.to_csv('bench_results_full_chunk.csv')

