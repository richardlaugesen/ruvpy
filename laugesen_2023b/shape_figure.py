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
import pandas as pd
from matplotlib import pyplot as plt
import time

from util import *

sys.path.append('..')

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *


def shape_figure(awrc, name, start_lt, end_lt, area, k_step=0.02, parallel_nodes=8, verbose=False):
    metadata = {
        'awrc': awrc,
        'name': name,
        'start_lt': start_lt,
        'end_lt': end_lt,
        'parallel_nodes': parallel_nodes,
        'k_step': k_step,
        'area': area
    }

    # load data
    obs, fcst, clim = load_data(awrc, start_lt, end_lt, area)

    print('Starting shape figure...')
    metadata['figure_name'] = 'shape'

    # generate results
    results = generate_results(obs, fcst, clim, k_step, parallel_nodes, verbose)

    # generate and save figure
    fig = generate_figure(results, metadata)
    save_figure(fig, metadata)

    # store all output
    output = {
        'obs': obs,
        'fcst': fcst,
        'clim': clim
    }
    output.update(metadata)
    output.update(results)
    save_results(output)

    return output


def generate_results(obs, fcst, ref, k_step, parallel_nodes, verbose):
    print('\tGenerating results')

    target_unity_risk_aversion = 0.3
    max_damages = 10000
    target_risk_premium = risk_aversion_coef_to_risk_premium(target_unity_risk_aversion, 1) 
    adjusted_risk_aversion = risk_premium_to_risk_aversion_coef(target_risk_premium, max_damages)

    # Define decision context
    decision_definition = {
        'alphas': np.array([0.1, 0.5, 0.9]),
        'target_unity_risk_aversion': target_unity_risk_aversion,
        'target_risk_premium': target_risk_premium,
        'adjusted_risk_aversion': adjusted_risk_aversion,
        'utility_function': [cara, {'A': adjusted_risk_aversion}],
        'economic_model': [cost_loss, cost_loss_analytical_spend],
        'decision_thresholds': None,
        'decision_method': 'optimise_over_forecast_distribution',
        'damage_function': [logistic, {'k': 1, 'A': max_damages, 'threshold': np.nanquantile(obs, 0.99)}]
    }

    # define range of logistic steepness parameters with more focus on low values
    # ks = np.exp(np.arange(0, 2, k_step)) - 1
    # ks = np.append(ks, [10, 50, 100]) # add few more values closer to a step function
    k_max = 5
    k_slope = 5 
    ks = (np.exp(np.arange(0, k_slope, k_step)) - 1) * k_max / (np.exp(k_slope - k_step) - 1)

    # generate streamflow-damage values for the different steepness for damage function figure
    select_ks = ks[[0, int(len(ks)/6), int(len(ks)/3), len(ks)-1]]
    streamflow = np.arange(0, np.nanmax(obs) * 1.3, 0.01)
    damage_fnc, params = decision_definition['damage_function']
    streamflow_damages = pd.DataFrame(index=streamflow, columns=select_ks)
    for k in select_ks:
        params['k'] = k
        streamflow_damages[k] = damage_fnc(params)(streamflow)

    # Calculate RUV for the different shape logistic damage functions for forecast value figure
    start_time = time.time()
    results = {}
    ruv_only = {}

    for i, k in enumerate(ks):
        progressor(i, len(ks), start_time)

        decision_definition['damage_function'] = [logistic, {'k': k, 'A': max_damages, 'threshold': np.nanquantile(obs, 0.99)}]
        results[k] = relative_utility_value(obs, fcst, ref, decision_definition, parallel_nodes=parallel_nodes, verbose=verbose)
        ruv_only[k] = results[k]['ruv']

    ruv_only = pd.DataFrame(ruv_only, index=decision_definition['alphas']).T    

    output = {
        'ruv_only': ruv_only, 
        'all_results': results, 
        'damages_results': streamflow_damages,
        'max_obs': np.nanmax(obs),
        'decision_definition': decision_definition,
        'steepness_params': ks,
        'execution_time_min': (time.time() - start_time) / 60
    }
    return output


def generate_figure(results, metadata):
    print('\tGenerating figure')

    fig, axes = create_panel()
    # fig.suptitle('Impact of damage function steepness, %s (%s) for lead-times %d to %d' 
    #              % (metadata['name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt']), 
    #              fontweight='semibold', fontsize='large')

    left_panel_color = LINE_COLORS['dark_blue']
    right_panel_color = LINE_COLORS['dark_orange']

    metadata['damages_title'] = 'Damage functions of different steepness'
    gen_damage_function_fig(results['damages_results'], metadata, 'k', results['max_obs'], axes[0], left_panel_color, LINE_STYLES)

    ax = axes[1]
    for i, column in enumerate(results['ruv_only'].columns):
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        ax.plot(results['ruv_only'].index, results['ruv_only'][column], 
                linewidth=1, alpha=1, color=right_panel_color, linestyle=line_style, 
                label=r'$\alpha$ = %.1f' % column)
 
    plt.axhline(0, color='grey', linewidth=0.5, alpha=0.3, linestyle='--', label='_hidden')

    plt.ylim(-1, 1)
    plt.xlim((0, 0.5))

    ax.set_xlabel('Logistic steepness parameter (k)')
    ax.set_ylabel('Forecast value (RUV)')
    ax.set_title(r'Forecast value for different values of $\alpha$', fontsize='medium')
    ax.legend()

    # Add labels to the top right corner of each panel
    axes[0].text(0.05, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=axes[0].transAxes)
    axes[1].text(0.05, 0.95, '(b)', horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)

    return fig