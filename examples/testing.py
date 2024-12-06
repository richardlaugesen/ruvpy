# Compare alternative decision types on a value diagram

# Copyright 2023 RUVPY Developers

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
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from ruvpy.relative_utility_value import relative_utility_value
from ruvpy.damage_functions import logistic_zero
from ruvpy.economic_models import cost_loss, cost_loss_analytical_spend
from ruvpy.utility_functions import cara, crra, power_value
from ruvpy.probability_weight_functions import power_weights, linear_weights
from ruvpy.decision_rules import optimise_over_forecast_distribution

# Load example forecast dataset

# Steamflow at Biggara in the Murray catchment of the southern Murray-Darling basin
# Subseasonal streamflow forecasts from MuTHRE and event frequency for reference

# load and convert runoff to cumecs
data = pd.read_csv('example_data/401012-muthre.csv.zip', index_col=0, parse_dates=True, dayfirst=True, compression='zip')
data *= 1165 / 86.4

# define weekly period sets
weeks = {
    'w1': (1, 7),
    'w2': (8, 14),
    'w3-4': (15, 30),
}

obs_sets = {}
fcst_sets = {}

# fetch the obs and forecast ensemble for each week
for week, (start, end) in weeks.items():
    curr = data[(data.index.day >= start) & (data.index.day <= end)]

    curr_obs = curr['obs']
    curr_fcst = curr[[col for col in curr.columns if col.startswith('ens-')]]
    
    # RUV library expects numpy arrays
    obs_sets[week] = curr_obs.values
    fcst_sets[week] = curr_fcst.values

    print(week, obs_sets[week].shape, fcst_sets[week].shape)

# Define decision context
parallel_nodes = 6
post_fix = 'polish_off'
alpha_step = 0.1
alphas = np.arange(alpha_step, 1, alpha_step)

# note there is no decision_thresholds defined, we will add this before calling relative_utility_value
decision_contexts = {}
decision_contexts['eut'] = {
    'damage_function': [logistic_zero, {'A': 1, 'k': 0.07, 'threshold': None}], # threshold defined later
    'utility_function': [cara, {'A': 1}],
    'economic_model': [cost_loss, cost_loss_analytical_spend, alphas],
    'decision_rule': [optimise_over_forecast_distribution, None],
    'polish': False
}

decision_contexts['cpt'] = decision_contexts['eut'].copy()
decision_contexts['cpt']['utility_function'] = [power_value, {'alpha': 0.6, 'beta': 0.6, 'lambda': 2}]
decision_contexts['cpt']['probability_weight_function'] = [power_weights, {'exponent': 0.65}]
decision_contexts['cpt']['reference_point'] = -0.3

ref = None  # tell RUV library to use obs to replicate event frequency reference as in REV

# Calculate RUV using different decision rules
results = xr.DataArray(np.nan, dims=('alpha', 'decision_type', 'week', 'decision_maker'), coords={'alpha': alphas, 'decision_type': ['binary', 'multi_categorical', 'continuous'], 'week': list(weeks.keys()), 'decision_maker': ['eut', 'cpt']})

for week in obs_sets.keys():
    for decision_context_key in ['cpt', 'eut']:
        decision_context = decision_contexts[decision_context_key]
        print(f"Calculating RUV for set {week} / {decision_context_key}")
    
        obs = obs_sets[week]
        fcst = fcst_sets[week]

        decision_context['damage_function'][1]['threshold'] = np.nanquantile(obs, 0.99)    
        print('Binary')
        decision_context['decision_thresholds'] = np.array([0, np.nanquantile(obs, 0.9)])
        results.loc[{'week': week, 'decision_type': 'binary', 'decision_maker': decision_context_key}] = relative_utility_value(obs, fcst, ref, decision_context, parallel_nodes)['ruv']

        print('Multi-categorical')
        decision_context['decision_thresholds'] = np.insert([np.nanquantile(obs, quant) for quant in [0.80, 0.85, 0.90, 0.95]], 0, 0)
        results.loc[{'week': week, 'decision_type': 'multi_categorical', 'decision_maker': decision_context_key}] = relative_utility_value(obs, fcst, ref, decision_context, parallel_nodes)['ruv']

        print('Continuous')
        decision_context['decision_thresholds'] = None
        results.loc[{'week': week, 'decision_type': 'continuous', 'decision_maker': decision_context_key}] = relative_utility_value(obs, fcst, ref, decision_context, parallel_nodes)['ruv']

results.to_netcdf(f'eut_cpt_test_results_{postfix}.nc')
        
# Plot results on a value diagram
weeks = results.coords['week'].values
decision_types = results.coords['decision_type'].values
alphas = results.coords['alpha'].values
decision_makers = results.coords['decision_maker'].values

for decision_maker in decision_makers:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, week in enumerate(weeks):
        ax = axs[i]
        for decision_type in decision_types:
            data = results.sel(week=week, decision_type=decision_type, decision_maker=decision_maker)
            ax.plot(alphas, data.values, label=decision_type)

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title('Forecast lead-times %s' % week)
        ax.set_xlabel(r'Relative expense of mitigation ($\alpha$)')
        if i == 0:
            ax.set_ylabel('Forecast value (RUV)')

        ax.legend()

    #import pdb; pdb.set_trace()
    fig.savefig(f"ruv_{decision_maker}_{postfix}.png")
    plt.close(fig)
