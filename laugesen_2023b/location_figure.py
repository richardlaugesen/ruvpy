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

import sys
import numpy as np
import pandas as pd
from scipy import stats
import time
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

sys.path.append('..')

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *
from util import *

def location_figure(awrc, name, start_lt, end_lt, area, q_step=0.2, parallel_nodes=8, verbose=False):
    metadata = {
        'awrc': awrc,
        'name': name,
        'start_lt': start_lt,
        'end_lt': end_lt,
        'parallel_nodes': parallel_nodes,
        'q_step': q_step,
        'area': area
    }

    # load data
    obs, fcst, clim = load_data(awrc, start_lt, end_lt, area)

    print('Starting location figure...')
    metadata['figure_name'] = 'location'

    # generate results
    results = generate_results(obs, fcst, clim, q_step, parallel_nodes, verbose)

    # generate and save figure
    fig = generate_figure(results, obs, metadata)
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


def generate_results(obs, fcst, ref, q_step, parallel_nodes, verbose):
    print('\tGenerating results')

    target_unity_risk_aversion = 0.3
    max_damages = 10000
    target_risk_premium = risk_aversion_coef_to_risk_premium(target_unity_risk_aversion, 1) 
    adjusted_risk_aversion = risk_premium_to_risk_aversion_coef(target_risk_premium, max_damages)

    # Define decision context
    decision_definition = {
        'econ_pars': np.array([0.1, 0.5, 0.9]),
        'target_unity_risk_aversion': target_unity_risk_aversion,
        'target_risk_premium': target_risk_premium,
        'adjusted_risk_aversion': adjusted_risk_aversion,
        'utility_function': [cara, {'A': adjusted_risk_aversion}],
        'economic_model': [cost_loss, cost_loss_analytical_spend],
        'decision_method': 'optimise_over_forecast_distribution',
        'decision_thresholds': None,
        'damage_function': [logistic, {'k': 0.2, 'A': max_damages, 'threshold': np.nanquantile(obs, 0.99)}]
    }

    thresholds = np.arange(0, np.nanmax(obs) * 1.3, q_step)
    
    # Calculate RUV for the different thresholds for damage function figure
    start_time = time.time()
    results = {}
    ruv_only = {}
    
    for i, threshold in enumerate(thresholds):
        progressor(i, len(thresholds), start_time)

        decision_definition['damage_function'][1]['threshold'] = threshold
        results[threshold] = relative_utility_value(obs, fcst, ref, decision_definition, parallel_nodes=parallel_nodes, verbose=verbose)
        ruv_only[threshold] = results[threshold]['ruv']

    ruv_only = pd.DataFrame(ruv_only, index=decision_definition['econ_pars']).T

    # Generate streamflow-damage values for the different thresholds for damage function figure
    select_thresholds = thresholds[[0, int(len(thresholds)/4), int(len(thresholds)/2), 3*int(len(thresholds)/4), len(thresholds)-1]]
    streamflow = np.arange(0, np.nanmax(obs) * 1.3, 0.01)
    damage_fnc, params = decision_definition['damage_function']
    streamflow_damages = pd.DataFrame(index=streamflow, columns=select_thresholds)
    for threshold in select_thresholds:
        params['threshold'] = threshold
        streamflow_damages[threshold] = damage_fnc(params)(streamflow)

    output = {
        'ruv_only': ruv_only, 
        'all_results': results, 
        'damages_results': streamflow_damages,
        'max_obs': np.nanmax(obs),
        'decision_definition': decision_definition,
        'thresholds': thresholds,
        'execution_time_min': (time.time() - start_time) / 60
    }
    return output


def generate_figure(results, obs, metadata, show_percentiles=True):
    print('\tGenerating figure')

    fig, axes = create_panel()
    # fig.suptitle('Impact of damage function location, %s (%s) for lead-times %d to %d' 
    #              % (metadata['name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt']), 
    #              fontweight='semibold', fontsize='large')

    left_panel_color = LINE_COLORS['dark_blue']
    right_panel_color = LINE_COLORS['dark_orange']

    metadata['damages_title'] = 'Damage functions of different locations'
    gen_damage_function_fig(results['damages_results'], metadata, r'$q_\tau$', results['max_obs'], axes[0], left_panel_color, LINE_STYLES)

    ax = axes[1]
    for i, column in enumerate(results['ruv_only'].columns):
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        ax.plot(results['ruv_only'].index, results['ruv_only'][column], 
                linewidth=1, alpha=1, color=right_panel_color, linestyle=line_style, 
                label=r'$\alpha$ = %.1f' % column)
 
    plt.ylim(0, 1)
    plt.xlim((0, 150))

    ax.set_xlabel(r'Damage function threshold $q_\tau$ ($m^3/s$)')
    ax.set_ylabel('Forecast value (RUV)')
    ax.set_title(r'Forecast value for different values of $\alpha$', fontsize='medium')
    ax.legend()

    # Add labels to the top right corner of each panel
    axes[0].text(0.05, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=axes[0].transAxes)
    axes[1].text(0.05, 0.95, '(b)', horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)

    def flow_to_percentile(x):
        return stats.percentileofscore(obs[~np.isnan(obs)], x)

    def percentile_to_flow(x):
        return np.nanpercentile(obs, x)

    if show_percentiles:
        secax = ax.secondary_xaxis(-0.15, functions=(flow_to_percentile, percentile_to_flow))
        secax.xaxis.set_major_locator(ticker.FixedLocator([50, 90, 99, 99.9, 100]))
        secax.set_xlabel('Percentile')

    return fig
